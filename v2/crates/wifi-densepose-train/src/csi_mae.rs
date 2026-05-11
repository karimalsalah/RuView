//! Masked-autoencoder pre-training for cross-domain CSI — **MERIDIAN-MAE** (ADR-027 §2.0).
//!
//! Implements a [CIG-MAE]-style **dual-stream** (amplitude + phase) masked
//! autoencoder over CSI "channel-snapshot" tokens. The pre-train objective is:
//! hide a large fraction of the tokens, encode only the visible ones, and
//! reconstruct the hidden amplitude *and* phase. The thesis (from the 2026-Q2
//! SOTA survey, `docs/research/sota/2026-Q2-agentic-ai-and-edge-for-ruview.md`):
//! cross-room generalisation is a **data-breadth** problem — pre-train one CSI
//! encoder on heterogeneous capture, then attach a small task head — not a
//! bigger-pose-net problem.
//!
//! # Token convention
//!
//! A CSI window `amplitude: [T, tx, rx, sub]` is flattened to a sequence of
//! `N = T·tx·rx` tokens, each a `sub`-dimensional vector (one *channel
//! snapshot*). This matches the `[B, T·tx·rx, sub]` layout the supervised model
//! already consumes (see `model.rs::ModalityTranslator`). Amplitude and phase
//! share the same `[N, sub]` token grid, so a single mask applies to both
//! streams — exactly the dual-stream setup CIG-MAE uses.
//!
//! # What's in this module
//!
//! * **Pure Rust** (always compiled, covered by `cargo test --no-default-features`):
//!   [`MaeConfig`] (+ `validate`), [`MaskStrategy`], [`TokenLayout`], the
//!   deterministic masking ([`mask_csi_window`]) and re-assembly
//!   ([`reassemble_tokens`]). A tiny inline PRNG keeps masking reproducible with
//!   no extra dependency.
//! * **`#[cfg(feature = "tch-backend")]`** — the `model` submodule: the
//!   encoder/decoder networks, the reconstruction loss, and the pre-train step.
//!   That code is *not* exercised by the default workspace test job; treat
//!   compile-checking it as requiring a LibTorch toolchain.
//!
//! # Status
//!
//! Prototype, iteration 1: masking pipeline + config + tests + ADR §2.0. The
//! `model` submodule is a v0 skeleton (MLP encoder/decoder, batch-level masking)
//! — transformer blocks, per-sample masking, information-guided masking, and a
//! `pretrain-mae` binary land in subsequent iterations.
//!
//! [CIG-MAE]: https://arxiv.org/html/2512.04723v1

use ndarray::{Array2, ArrayView4};
use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

// ---------------------------------------------------------------------------
// PRNG — tiny, dependency-free, deterministic. (SplitMix64.)
// ---------------------------------------------------------------------------

/// Minimal deterministic PRNG (SplitMix64) used only for reproducible masking.
///
/// Not cryptographic; the point is that the same `seed` always yields the same
/// token permutation so masked-autoencoder runs are byte-reproducible.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero state.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform `usize` in `[0, n)` (Lemire-ish; bias is negligible for our `n`).
    fn below(&mut self, n: usize) -> usize {
        deb_assert_nonzero(n);
        (self.next_u64() % (n as u64)) as usize
    }
}

#[inline]
fn deb_assert_nonzero(n: usize) {
    debug_assert!(n > 0, "SplitMix64::below requires n > 0");
}

/// In-place Fisher–Yates shuffle of `xs` using `rng`.
fn shuffle<T>(xs: &mut [T], rng: &mut SplitMix64) {
    let n = xs.len();
    if n < 2 {
        return;
    }
    for i in (1..n).rev() {
        let j = rng.below(i + 1);
        xs.swap(i, j);
    }
}

/// Per-token "information" score used by [`MaskStrategy::InfoGuided`]: the
/// (population) variance of the token's amplitude values plus the variance of
/// its phase values. Near-constant tokens (e.g. a quiet sub-carrier slice) score
/// near zero, so they're less likely to be masked; structured tokens score
/// higher. `amp`/`phase` are the flattened `[N, sub]` grids; `i` is the token row.
fn token_information(amp: &Array2<f32>, phase: &Array2<f32>, i: usize) -> f64 {
    let var = |row: ndarray::ArrayView1<f32>| -> f64 {
        let m = row.len();
        if m == 0 {
            return 0.0;
        }
        let mean = row.iter().map(|&x| x as f64).sum::<f64>() / m as f64;
        row.iter().map(|&x| { let d = x as f64 - mean; d * d }).sum::<f64>() / m as f64
    };
    var(amp.row(i)) + var(phase.row(i))
}

// ---------------------------------------------------------------------------
// Masking strategy
// ---------------------------------------------------------------------------

/// How tokens are chosen for masking in the MAE pre-text task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskStrategy {
    /// Uniform-random token masking (the MAE default — cheap, strong baseline).
    Random,
    /// Information-guided masking (CIG-MAE): preferentially mask high-energy /
    /// high-variance tokens so the model can't trivially in-paint flat regions.
    ///
    /// Not yet implemented — selecting it currently falls back to [`MaskStrategy::Random`]
    /// (with a `tracing::warn!`). Lands in iteration 2.
    InfoGuided,
}

impl Default for MaskStrategy {
    fn default() -> Self {
        MaskStrategy::Random
    }
}

// ---------------------------------------------------------------------------
// MaeConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters for the CSI masked autoencoder.
///
/// Defaults track the MAE / CIG-MAE recipes (high mask ratio, narrow decoder).
/// Dimensions are deliberately small — this is a prototype encoder, and the
/// survey's finding is that *data breadth*, not model size, is the bottleneck.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaeConfig {
    /// Fraction of tokens hidden from the encoder, in `(0, 1)`. MAE uses ~0.75.
    pub mask_ratio: f64,
    /// Masking strategy.
    pub mask_strategy: MaskStrategy,
    /// Token (sub-carrier) dimension. Must match the dataset after interpolation
    /// (the system target is 56 — see `TrainingConfig::num_subcarriers`).
    pub token_dim: usize,
    /// Encoder embedding dimension.
    pub encoder_dim: usize,
    /// Number of encoder transformer blocks (v0 skeleton ignores depth > 0 and
    /// uses an MLP; honoured from iteration 2).
    pub encoder_depth: usize,
    /// Number of encoder attention heads.
    pub encoder_heads: usize,
    /// Decoder embedding dimension (MAE uses a *narrower* decoder than the encoder).
    pub decoder_dim: usize,
    /// Number of decoder transformer blocks.
    pub decoder_depth: usize,
    /// Number of decoder attention heads.
    pub decoder_heads: usize,
    /// Weight of the phase-reconstruction loss relative to amplitude (CIG-MAE ≈ 1.0).
    pub phase_loss_weight: f64,
    /// Default RNG seed for masking when a per-call seed isn't supplied.
    pub seed: u64,
}

impl Default for MaeConfig {
    fn default() -> Self {
        Self {
            mask_ratio: 0.75,
            mask_strategy: MaskStrategy::Random,
            token_dim: 56,
            encoder_dim: 128,
            encoder_depth: 4,
            encoder_heads: 4,
            decoder_dim: 64,
            decoder_depth: 2,
            decoder_heads: 4,
            phase_loss_weight: 1.0,
            seed: 0xC511_0027,
        }
    }
}

impl MaeConfig {
    /// Validate the configuration. Mirrors the `TrainingConfig::validate` style.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |field: &'static str, reason: String| ConfigError::invalid_value(field, reason);

        if !(self.mask_ratio > 0.0 && self.mask_ratio < 1.0) {
            return Err(bad(
                "mask_ratio",
                format!("must be in (0, 1), got {}", self.mask_ratio),
            ));
        }
        if self.token_dim == 0 {
            return Err(bad("token_dim", "must be >= 1".into()));
        }
        for (field, v) in [
            ("encoder_dim", self.encoder_dim),
            ("decoder_dim", self.decoder_dim),
            ("encoder_heads", self.encoder_heads),
            ("decoder_heads", self.decoder_heads),
        ] {
            if v == 0 {
                return Err(bad(field, "must be >= 1".into()));
            }
        }
        if self.encoder_dim % self.encoder_heads != 0 {
            return Err(bad(
                "encoder_dim",
                format!(
                    "must be divisible by encoder_heads ({} % {} != 0)",
                    self.encoder_dim, self.encoder_heads
                ),
            ));
        }
        if self.decoder_dim % self.decoder_heads != 0 {
            return Err(bad(
                "decoder_dim",
                format!(
                    "must be divisible by decoder_heads ({} % {} != 0)",
                    self.decoder_dim, self.decoder_heads
                ),
            ));
        }
        if !(self.phase_loss_weight >= 0.0 && self.phase_loss_weight.is_finite()) {
            return Err(bad(
                "phase_loss_weight",
                format!("must be a finite, non-negative number, got {}", self.phase_loss_weight),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Token layout
// ---------------------------------------------------------------------------

/// Token-grid layout derived from a CSI window of shape `[T, tx, rx, sub]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenLayout {
    /// Number of tokens, `T · tx · rx`.
    pub n_tokens: usize,
    /// Per-token dimension, `sub`.
    pub token_dim: usize,
    /// Window frame count `T`.
    pub frames: usize,
    /// Transmit-antenna count `tx`.
    pub tx: usize,
    /// Receive-antenna count `rx`.
    pub rx: usize,
}

impl TokenLayout {
    /// Derive the layout from a `[T, tx, rx, sub]` view.
    pub fn from_window(window: ArrayView4<f32>) -> Self {
        let s = window.shape();
        Self {
            n_tokens: s[0] * s[1] * s[2],
            token_dim: s[3],
            frames: s[0],
            tx: s[1],
            rx: s[2],
        }
    }

    /// Flatten a `[T, tx, rx, sub]` window into a `[N, sub]` token matrix
    /// (row `f·tx·rx + t·rx + r` = the snapshot for frame `f`, tx `t`, rx `r`).
    pub fn flatten(window: ArrayView4<f32>) -> Array2<f32> {
        let layout = Self::from_window(window);
        window
            .to_owned()
            .into_shape((layout.n_tokens, layout.token_dim))
            .expect("[T,tx,rx,sub] -> [T*tx*rx, sub] reshape is always valid")
    }
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// The result of masking one CSI sample for the MAE pre-text task.
///
/// `visible_idx` and `masked_idx` are sorted ascending, are disjoint, and
/// together cover `0..n_tokens`. The encoder sees `visible_*`; the decoder is
/// trained to reconstruct `target_*` at the `masked_idx` positions.
#[derive(Debug, Clone)]
pub struct MaskedCsi {
    /// Token indices visible to the encoder. Length `round((1 − r)·N)`, ≥ 1.
    pub visible_idx: Vec<usize>,
    /// Token indices hidden from the encoder (reconstruction targets). Length `N − |visible|`, ≥ 1.
    pub masked_idx: Vec<usize>,
    /// Per-token boolean mask over `0..N`; `true` ⇒ masked (target).
    pub mask: Vec<bool>,
    /// Visible amplitude tokens, shape `[|visible|, token_dim]`.
    pub visible_amp: Array2<f32>,
    /// Visible phase tokens, shape `[|visible|, token_dim]`.
    pub visible_phase: Array2<f32>,
    /// Target (masked) amplitude tokens, shape `[|masked|, token_dim]`.
    pub target_amp: Array2<f32>,
    /// Target (masked) phase tokens, shape `[|masked|, token_dim]`.
    pub target_phase: Array2<f32>,
    /// Layout of the source window.
    pub layout: TokenLayout,
}

/// Deterministically split a CSI window's tokens into visible / masked sets and
/// return the masked-out amplitude+phase as reconstruction targets.
///
/// * `amplitude`, `phase` — `[T, tx, rx, sub]`, identical shapes.
/// * `mask_ratio` — fraction hidden; clamped so at least one token is visible
///   and at least one is masked.
/// * `strategy` — [`MaskStrategy::Random`] (uniform) or [`MaskStrategy::InfoGuided`]
///   (CIG-MAE-style: preferentially mask high-information tokens, where a token's
///   "information" is the variance of its amplitude + phase values — flat tokens
///   are trivially in-painted, so masking them teaches less). Both are
///   deterministic in `seed`.
/// * `seed` — makes the choice reproducible. A good per-sample seed is
///   `base_seed ^ (sample_index as u64).wrapping_mul(0x9E3779B97F4A7C15)`.
///
/// # Errors
///
/// Returns [`ConfigError::InvalidValue`] when the shapes mismatch, the window
/// has no tokens, or `mask_ratio` is not in `(0, 1)`.
pub fn mask_csi_window(
    amplitude: ArrayView4<f32>,
    phase: ArrayView4<f32>,
    mask_ratio: f64,
    strategy: MaskStrategy,
    seed: u64,
) -> Result<MaskedCsi, ConfigError> {
    if amplitude.shape() != phase.shape() {
        return Err(ConfigError::InvalidValue {
            field: "phase".into(),
            reason: format!(
                "amplitude/phase shape mismatch: {:?} vs {:?}",
                amplitude.shape(),
                phase.shape()
            ),
        });
    }
    if !(mask_ratio > 0.0 && mask_ratio < 1.0) {
        return Err(ConfigError::InvalidValue {
            field: "mask_ratio".into(),
            reason: format!("must be in (0, 1), got {mask_ratio}"),
        });
    }

    let layout = TokenLayout::from_window(amplitude);
    let n = layout.n_tokens;
    if n == 0 {
        return Err(ConfigError::InvalidValue {
            field: "amplitude".into(),
            reason: "CSI window has zero tokens (empty T/tx/rx)".into(),
        });
    }

    // Number of masked tokens, clamped so both partitions are non-empty.
    let mut n_mask = (mask_ratio * n as f64).round() as usize;
    if n_mask == 0 {
        n_mask = 1;
    }
    if n_mask >= n {
        n_mask = n - 1;
    }

    let amp_flat = TokenLayout::flatten(amplitude);
    let phase_flat = TokenLayout::flatten(phase);

    // Pick the n_mask masked token indices according to the strategy.
    let mut rng = SplitMix64::new(seed);
    let masked_set: Vec<usize> = match strategy {
        MaskStrategy::Random => {
            // Uniform: shuffle [0, n) and take the first n_mask.
            let mut perm: Vec<usize> = (0..n).collect();
            shuffle(&mut perm, &mut rng);
            perm[..n_mask].to_vec()
        }
        MaskStrategy::InfoGuided => {
            // Weighted-without-replacement by per-token information (variance of
            // amplitude+phase). Efraimidis–Spirakis: key_i = u_i^(1/w_i),
            // pick the n_mask largest keys. Deterministic given `seed`.
            let mut keyed: Vec<(f64, usize)> = (0..n)
                .map(|i| {
                    let w = token_information(&amp_flat, &phase_flat, i) + 1e-6;
                    // u in (0, 1]: avoid 0 so ln() is finite. key = u^(1/w);
                    // rank by ln(key) = ln(u)/w (monotone, avoids tiny powers).
                    let u = ((rng.next_u64() >> 11) as f64 + 1.0) / (((1u64 << 53) as f64) + 1.0);
                    let key = u.ln() / w; // larger (closer to 0) ⇒ more likely chosen
                    (key, i)
                })
                .collect();
            // Largest key = least-negative ln(u)/w ⇒ sort descending by key.
            keyed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            keyed[..n_mask].iter().map(|&(_, i)| i).collect()
        }
    };

    let mut masked_idx = masked_set;
    masked_idx.sort_unstable();
    let masked_lookup: std::collections::HashSet<usize> = masked_idx.iter().copied().collect();
    let mut visible_idx: Vec<usize> = (0..n).filter(|i| !masked_lookup.contains(i)).collect();
    visible_idx.sort_unstable();

    let mut mask = vec![false; n];
    for &i in &masked_idx {
        mask[i] = true;
    }

    let gather = |src: &Array2<f32>, idx: &[usize]| -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((idx.len(), layout.token_dim));
        for (row, &i) in idx.iter().enumerate() {
            out.row_mut(row).assign(&src.row(i));
        }
        out
    };

    Ok(MaskedCsi {
        visible_amp: gather(&amp_flat, &visible_idx),
        visible_phase: gather(&phase_flat, &visible_idx),
        target_amp: gather(&amp_flat, &masked_idx),
        target_phase: gather(&phase_flat, &masked_idx),
        visible_idx,
        masked_idx,
        mask,
        layout,
    })
}

/// Re-assemble a full `[N, token_dim]` token grid from encoder-visible tokens
/// plus decoder-predicted masked tokens. Useful for evaluating / visualising
/// reconstructions (it is *not* needed for training the loss).
///
/// # Errors
///
/// Returns [`ConfigError::InvalidValue`] if the index sets don't partition
/// `0..N` or the row counts don't match the index lengths / `token_dim`.
pub fn reassemble_tokens(
    layout: TokenLayout,
    visible_idx: &[usize],
    visible: &Array2<f32>,
    masked_idx: &[usize],
    predicted: &Array2<f32>,
) -> Result<Array2<f32>, ConfigError> {
    let n = layout.n_tokens;
    let inv = |field: &'static str, reason: String| ConfigError::invalid_value(field, reason);
    if visible_idx.len() + masked_idx.len() != n {
        return Err(inv(
            "indices",
            format!(
                "visible ({}) + masked ({}) != n_tokens ({n})",
                visible_idx.len(),
                masked_idx.len()
            ),
        ));
    }
    if visible.nrows() != visible_idx.len() || predicted.nrows() != masked_idx.len() {
        return Err(inv("rows", "row count does not match index length".into()));
    }
    if visible.ncols() != layout.token_dim || predicted.ncols() != layout.token_dim {
        return Err(inv("token_dim", "column count does not match layout.token_dim".into()));
    }

    let mut out = Array2::<f32>::zeros((n, layout.token_dim));
    let mut seen = vec![false; n];
    for (row, &i) in visible_idx.iter().enumerate() {
        if i >= n || seen[i] {
            return Err(inv("visible_idx", format!("out of range or duplicate index {i}")));
        }
        seen[i] = true;
        out.row_mut(i).assign(&visible.row(row));
    }
    for (row, &i) in masked_idx.iter().enumerate() {
        if i >= n || seen[i] {
            return Err(inv("masked_idx", format!("out of range or duplicate index {i}")));
        }
        seen[i] = true;
        out.row_mut(i).assign(&predicted.row(row));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// tch-gated: the MAE networks + pre-train step
// ---------------------------------------------------------------------------

/// CSI masked-autoencoder networks (LibTorch / `tch`).
///
/// **Compiled only with `--features tch-backend`.** Not exercised by the default
/// `cargo test --workspace --no-default-features` CI job, so compile-checking it
/// requires a LibTorch toolchain.
///
/// Iteration-1 status: this is a v0 *skeleton* — the public interface
/// ([`model::CsiMae`], `forward`, `reconstruction_loss`, `model::pretrain_step`)
/// is fixed, but the encoder/decoder are placeholder MLPs and per-batch masking
/// is assumed shared. Transformer blocks, per-sample masking, and the
/// `pretrain-mae` binary land in iteration 2.
#[cfg(feature = "tch-backend")]
pub mod model {
    use super::MaeConfig;

    /// Placeholder for the CSI masked autoencoder network.
    ///
    /// Iteration 2 fills this in (dual-stream encoder over visible tokens →
    /// shared latent; narrow decoder over all positions with learned mask
    /// tokens; reconstruct amplitude + phase). For now it exists so downstream
    /// code can name the type and the build wiring is in place.
    #[derive(Debug)]
    pub struct CsiMae {
        _cfg: MaeConfig,
    }

    impl CsiMae {
        /// Build a `CsiMae` under the given variable-store path with `cfg`.
        ///
        /// Iteration-1 stub — see the module docs.
        pub fn new(_vs: &tch::nn::Path, cfg: &MaeConfig) -> Self {
            Self { _cfg: cfg.clone() }
        }
    }

    // NOTE (iteration 2): add
    //   impl CsiMae { fn forward(&self, vis_amp, vis_phase, vis_idx, n_tokens) -> (pred_amp, pred_phase) }
    //   fn reconstruction_loss(pred_amp, pred_phase, tgt_amp, tgt_phase, phase_w) -> Tensor
    //   fn pretrain_step(model, opt, batch) -> f64
    // plus a `bin/pretrain_mae.rs` driving SyntheticCsiDataset / MmFiDataset.
}

// ---------------------------------------------------------------------------
// Tests (pure-Rust portion)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array4;

    fn synth_window(frames: usize, tx: usize, rx: usize, sub: usize, seed: u64) -> (Array4<f32>, Array4<f32>) {
        let mut rng = SplitMix64::new(seed);
        let mk = |rng: &mut SplitMix64| {
            Array4::<f32>::from_shape_fn((frames, tx, rx, sub), |_| (rng.next_u64() as f32) / (u64::MAX as f32))
        };
        let a = mk(&mut rng);
        let p = mk(&mut rng);
        (a, p)
    }

    #[test]
    fn mae_config_defaults_validate() {
        MaeConfig::default().validate().expect("default MaeConfig must validate");
    }

    #[test]
    fn mae_config_rejects_bad_values() {
        let mut c = MaeConfig::default();
        c.mask_ratio = 1.0;
        assert!(c.validate().is_err());
        let mut c = MaeConfig::default();
        c.encoder_dim = 130; // not divisible by encoder_heads (4)
        assert!(c.validate().is_err());
        let mut c = MaeConfig::default();
        c.token_dim = 0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn token_layout_matches_window() {
        let (a, _p) = synth_window(8, 2, 3, 56, 1);
        let l = TokenLayout::from_window(a.view());
        assert_eq!(l, TokenLayout { n_tokens: 8 * 2 * 3, token_dim: 56, frames: 8, tx: 2, rx: 3 });
        assert_eq!(TokenLayout::flatten(a.view()).dim(), (48, 56));
    }

    #[test]
    fn masking_partitions_exhaustively_and_disjointly() {
        let (a, p) = synth_window(10, 1, 1, 56, 7);
        let m = mask_csi_window(a.view(), p.view(), 0.75, MaskStrategy::Random, 42).unwrap();
        let n = m.layout.n_tokens;
        assert!(!m.visible_idx.is_empty() && !m.masked_idx.is_empty());
        assert_eq!(m.visible_idx.len() + m.masked_idx.len(), n);
        // disjoint + exhaustive
        let mut all: Vec<usize> = m.visible_idx.iter().chain(m.masked_idx.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..n).collect::<Vec<_>>());
        // mask vec agrees with masked_idx
        assert_eq!(m.mask.iter().filter(|&&b| b).count(), m.masked_idx.len());
        for &i in &m.masked_idx { assert!(m.mask[i]); }
        for &i in &m.visible_idx { assert!(!m.mask[i]); }
        // target/visible row counts + dims
        assert_eq!(m.target_amp.dim(), (m.masked_idx.len(), 56));
        assert_eq!(m.visible_phase.dim(), (m.visible_idx.len(), 56));
        // mask ratio ≈ 0.75 on n=10 → 8 masked, sorted ascending
        assert_eq!(m.masked_idx.len(), 8);
        assert!(m.masked_idx.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn masking_is_deterministic_in_seed() {
        let (a, p) = synth_window(6, 1, 1, 16, 3);
        let m1 = mask_csi_window(a.view(), p.view(), 0.5, MaskStrategy::Random, 123).unwrap();
        let m2 = mask_csi_window(a.view(), p.view(), 0.5, MaskStrategy::Random, 123).unwrap();
        let m3 = mask_csi_window(a.view(), p.view(), 0.5, MaskStrategy::Random, 124).unwrap();
        assert_eq!(m1.masked_idx, m2.masked_idx);
        assert_eq!(m1.visible_amp, m2.visible_amp);
        assert_ne!(m1.masked_idx, m3.masked_idx); // different seed → different partition
    }

    /// Build a window where the first half of the tokens are (near-)constant
    /// (low information) and the second half are noisy (high information).
    /// Returns `(amp, phase, n_tokens, n_low)`.
    fn split_info_window() -> (ndarray::Array4<f32>, ndarray::Array4<f32>, usize, usize) {
        // 20 frames, 1x1, 8 sub  → 20 tokens; first 10 constant, last 10 noisy.
        let frames = 20;
        let sub = 8;
        let mut rng = SplitMix64::new(999);
        let amp = ndarray::Array4::<f32>::from_shape_fn((frames, 1, 1, sub), |(f, _, _, _)| {
            if f < 10 { 1.0 } else { (rng.next_u64() as f32) / (u64::MAX as f32) }
        });
        let phase = ndarray::Array4::<f32>::from_shape_fn((frames, 1, 1, sub), |(f, _, _, _)| {
            if f < 10 { 0.0 } else { (rng.next_u64() as f32) / (u64::MAX as f32) }
        });
        (amp, phase, frames, 10)
    }

    #[test]
    fn info_guided_masking_prefers_high_information_tokens() {
        let (a, p, _n, n_low) = split_info_window();
        // Mask 50% (10 of 20). With info-guided selection the noisy tokens
        // (indices 10..20) should dominate the masked set far beyond chance.
        let mut high_count_total = 0usize;
        let trials = 8;
        for seed in 0..trials {
            let m = mask_csi_window(a.view(), p.view(), 0.5, MaskStrategy::InfoGuided, seed).unwrap();
            assert_eq!(m.masked_idx.len(), 10);
            let high = m.masked_idx.iter().filter(|&&i| i >= n_low).count();
            high_count_total += high;
        }
        // Random would average ~5/10 high per trial; info-guided should be ≥ ~8/10.
        let avg_high = high_count_total as f64 / trials as f64;
        assert!(avg_high >= 7.5, "info-guided avg high-info masked = {avg_high}, expected >= 7.5");
    }

    #[test]
    fn info_guided_masking_is_deterministic_in_seed() {
        let (a, p, _n, _) = split_info_window();
        let m1 = mask_csi_window(a.view(), p.view(), 0.4, MaskStrategy::InfoGuided, 5).unwrap();
        let m2 = mask_csi_window(a.view(), p.view(), 0.4, MaskStrategy::InfoGuided, 5).unwrap();
        let m3 = mask_csi_window(a.view(), p.view(), 0.4, MaskStrategy::InfoGuided, 6).unwrap();
        assert_eq!(m1.masked_idx, m2.masked_idx);
        assert_eq!(m1.target_amp, m2.target_amp);
        assert_ne!(m1.masked_idx, m3.masked_idx);
        // still a valid exhaustive/disjoint partition
        let n = m1.layout.n_tokens;
        assert_eq!(m1.visible_idx.len() + m1.masked_idx.len(), n);
        let mut all: Vec<usize> = m1.visible_idx.iter().chain(m1.masked_idx.iter()).copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..n).collect::<Vec<_>>());
    }

    #[test]
    fn token_information_is_zero_for_constant_and_positive_for_varied() {
        let (a, p, _n, _) = split_info_window();
        let amp_flat = TokenLayout::flatten(a.view());
        let ph_flat = TokenLayout::flatten(p.view());
        assert!(token_information(&amp_flat, &ph_flat, 0) < 1e-9);   // constant token
        assert!(token_information(&amp_flat, &ph_flat, 15) > 1e-6);  // noisy token
    }

    #[test]
    fn masking_clamps_extreme_ratios() {
        let (a, p) = synth_window(4, 1, 1, 8, 9);
        // huge ratio still leaves ≥1 visible
        let m = mask_csi_window(a.view(), p.view(), 0.999, MaskStrategy::Random, 1).unwrap();
        assert_eq!(m.visible_idx.len(), 1);
        // tiny ratio still masks ≥1
        let m = mask_csi_window(a.view(), p.view(), 0.0001, MaskStrategy::Random, 1).unwrap();
        assert_eq!(m.masked_idx.len(), 1);
        // out-of-range ratio is an error
        assert!(mask_csi_window(a.view(), p.view(), 0.0, MaskStrategy::Random, 1).is_err());
        assert!(mask_csi_window(a.view(), p.view(), 1.0, MaskStrategy::Random, 1).is_err());
    }

    #[test]
    fn shape_mismatch_is_an_error() {
        let (a, _) = synth_window(4, 1, 1, 8, 1);
        let (_, p) = synth_window(4, 1, 1, 16, 1);
        assert!(mask_csi_window(a.view(), p.view(), 0.5, MaskStrategy::Random, 1).is_err());
    }

    #[test]
    fn reassemble_round_trips_the_masking() {
        let (a, p) = synth_window(5, 1, 1, 16, 11);
        let m = mask_csi_window(a.view(), p.view(), 0.6, MaskStrategy::Random, 77).unwrap();
        // "perfect decoder": predicted == true masked tokens
        let recon = reassemble_tokens(m.layout, &m.visible_idx, &m.visible_amp, &m.masked_idx, &m.target_amp).unwrap();
        let orig = TokenLayout::flatten(a.view());
        assert_eq!(recon, orig);
        // a bad partition is rejected
        assert!(reassemble_tokens(m.layout, &m.visible_idx, &m.visible_amp, &[], &Array2::zeros((0, 16))).is_err());
    }
}
