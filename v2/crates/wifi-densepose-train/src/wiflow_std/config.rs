//! Configuration and pure-Rust shape/parameter math for WiFlow-STD
//! (ADR-152 §2.2). See the [module docs](crate::wiflow_std) for provenance.
//!
//! Everything here compiles without the `tch-backend` feature so the
//! architecture's invariants (parameter count, output shapes, divisibility
//! constraints) are unit-testable under `--no-default-features`. The
//! 15-keypoint default must yield exactly **2,225,042** parameters — the
//! count verified against the upstream reference (`RESULTS.md`).

use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// TCN kernel size — fixed at 3 in the reference architecture.
pub const TCN_KERNEL: usize = 3;

/// Dropout used inside the 2-D conv blocks (`Dropout2d`). The reference
/// hardcodes 0.3 in `convnet.py` (the model-level `dropout` argument is only
/// forwarded to the TCN), so it is a constant here rather than a config field.
pub const CONV_BLOCK_DROPOUT: f64 = 0.3;

// ---------------------------------------------------------------------------
// WiFlowStdConfig
// ---------------------------------------------------------------------------

/// Hyper-parameters for the WiFlow-STD pose model (ADR-152 §2.2).
///
/// Defaults reproduce the verified upstream architecture exactly (2,225,042
/// parameters, 15 keypoints). For RuView's ESP32 17-keypoint eval set
/// (ADR-152 §2.2(b)) use [`WiFlowStdConfig::for_keypoints`]`(17)` — the
/// keypoint count only changes the final adaptive pooling, not the parameter
/// count, so retrained 15-keypoint weights remain shape-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WiFlowStdConfig {
    /// CSI input feature dimension (subcarriers × antenna paths flattened).
    /// Must be divisible by [`Self::tcn_groups`]. Default: **540**.
    pub subcarriers: usize,

    /// Temporal window length in CSI frames. Default: **20**.
    pub window: usize,

    /// Output channels of each TCN level (dilation doubles per level:
    /// 1, 2, 4, 8, …). Every entry must be divisible by [`Self::tcn_groups`].
    /// Default: **[540, 440, 340, 240]** — the `models/` code values, *not*
    /// upstream `config.py`'s stale `[480, 360, 240]`.
    pub tcn_channels: Vec<usize>,

    /// Group count for the depthwise-grouped TCN convolutions. The reference
    /// hardcodes **20**; exposed so non-540 subcarrier layouts can keep the
    /// divisibility invariant. Default: **20**.
    pub tcn_groups: usize,

    /// Output channels of the 2-D conv encoder blocks. The first entry is
    /// also `ConvBlock1`'s output; each subsequent block downsamples the
    /// subcarrier axis by 2. Default: **[8, 16, 32, 64]**.
    pub conv_channels: Vec<usize>,

    /// Attention head groups for the dual axial attention. Must divide the
    /// last entry of [`Self::conv_channels`]. Default: **8**.
    pub attention_groups: usize,

    /// Number of 2-D keypoints produced. Default: **15** (upstream skeleton);
    /// use **17** for RuView's COCO-skeleton ESP32 eval set.
    pub keypoints: usize,

    /// Elementwise dropout probability inside the TCN blocks, in `[0, 1)`.
    /// Default: **0.5** (the value used by our verified retraining run).
    pub dropout: f64,
}

impl Default for WiFlowStdConfig {
    fn default() -> Self {
        WiFlowStdConfig {
            subcarriers: 540,
            window: 20,
            tcn_channels: vec![540, 440, 340, 240],
            tcn_groups: 20,
            conv_channels: vec![8, 16, 32, 64],
            attention_groups: 8,
            keypoints: 15,
            dropout: 0.5,
        }
    }
}

impl WiFlowStdConfig {
    /// Default architecture with a different keypoint count (e.g. 17 for the
    /// ESP32 COCO-skeleton eval set, ADR-152 §2.2(b)).
    pub fn for_keypoints(keypoints: usize) -> Self {
        WiFlowStdConfig {
            keypoints,
            ..Self::default()
        }
    }

    /// Validate all architectural invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::InvalidValue`] naming the offending field.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.subcarriers == 0 {
            return Err(ConfigError::invalid_value("subcarriers", "must be >= 1"));
        }
        if self.window == 0 {
            return Err(ConfigError::invalid_value("window", "must be >= 1"));
        }
        if self.tcn_groups == 0 {
            return Err(ConfigError::invalid_value("tcn_groups", "must be >= 1"));
        }
        if self.subcarriers % self.tcn_groups != 0 {
            return Err(ConfigError::invalid_value(
                "subcarriers",
                format!(
                    "{} is not divisible by tcn_groups={} (grouped conv requirement)",
                    self.subcarriers, self.tcn_groups
                ),
            ));
        }
        if self.tcn_channels.is_empty() {
            return Err(ConfigError::invalid_value(
                "tcn_channels",
                "must contain at least one level",
            ));
        }
        for (i, &c) in self.tcn_channels.iter().enumerate() {
            if c == 0 || c % self.tcn_groups != 0 {
                return Err(ConfigError::invalid_value(
                    "tcn_channels",
                    format!(
                        "level {i} has {c} channels; must be > 0 and divisible by tcn_groups={}",
                        self.tcn_groups
                    ),
                ));
            }
        }
        if self.conv_channels.is_empty() {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                "must contain at least one block",
            ));
        }
        if self.conv_channels.iter().any(|&c| c == 0) {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                "all blocks must have > 0 channels",
            ));
        }
        let c_last = *self.conv_channels.last().expect("non-empty checked above");
        if self.attention_groups == 0 || c_last % self.attention_groups != 0 {
            return Err(ConfigError::invalid_value(
                "attention_groups",
                format!(
                    "{} must be >= 1 and divide the last conv channel count {c_last}",
                    self.attention_groups
                ),
            ));
        }
        if c_last < 2 || c_last % 2 != 0 {
            return Err(ConfigError::invalid_value(
                "conv_channels",
                format!("last block has {c_last} channels; decoder needs an even count >= 2"),
            ));
        }
        if self.keypoints == 0 {
            return Err(ConfigError::invalid_value("keypoints", "must be >= 1"));
        }
        if !self.dropout.is_finite() || !(0.0..1.0).contains(&self.dropout) {
            return Err(ConfigError::invalid_value(
                "dropout",
                format!("{} is outside [0, 1)", self.dropout),
            ));
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shape inference
    // -----------------------------------------------------------------------

    /// Channel count produced by the TCN stack (last TCN level). This is the
    /// *width* of the image-like tensor fed to the 2-D encoder.
    pub fn tcn_output_channels(&self) -> usize {
        *self.tcn_channels.last().unwrap_or(&0)
    }

    /// Width of the encoder feature map after the strided conv blocks.
    ///
    /// `ConvBlock1` preserves width; each `AsymmetricConvBlock` applies a
    /// `(1, 3)` kernel with stride `(1, 2)` and padding `(0, 1)`:
    /// `w → (w - 1) / 2 + 1`. Default: 240 → 120 → 60 → 30 → **15**.
    pub fn feature_width(&self) -> usize {
        let mut w = self.tcn_output_channels();
        for _ in &self.conv_channels {
            w = (w.saturating_sub(1)) / 2 + 1;
        }
        w
    }

    /// Output tensor shape `(batch, keypoints, 2)`. The adaptive average pool
    /// maps the feature height to `keypoints` regardless of its size, so the
    /// keypoint count is free (15 and 17 share identical weights).
    pub fn output_shape(&self, batch: usize) -> (usize, usize, usize) {
        (batch, self.keypoints, 2)
    }

    // -----------------------------------------------------------------------
    // Parameter-count formula
    // -----------------------------------------------------------------------

    /// Total trainable parameter count, derived layer-by-layer from the
    /// architecture (BatchNorm weight+bias counted; running stats are buffers
    /// and excluded, matching PyTorch's `numel` convention).
    ///
    /// Pins the port against the verified reference: the 15-keypoint default
    /// must equal **2,225,042** (`RESULTS.md` artifact verification).
    pub fn param_count(&self) -> usize {
        let mut total = 0;

        // TCN stack.
        let mut c_in = self.subcarriers;
        for &c_out in &self.tcn_channels {
            total += tcn_block_params(c_in, c_out, TCN_KERNEL, self.tcn_groups);
            c_in = c_out;
        }

        // ConvBlock1 (1 → conv_channels[0]) + asymmetric blocks. Both block
        // kinds have identical parameter shapes (stride changes nothing).
        let mut c_in = 1;
        total += conv_block_params(c_in, self.conv_channels[0]);
        c_in = self.conv_channels[0];
        for &c_out in &self.conv_channels {
            total += conv_block_params(c_in, c_out);
            c_in = c_out;
        }

        // Dual axial attention: width axis + height axis, both c_in → c_in.
        total += 2 * axial_attention_params(c_in, self.attention_groups);

        // Decoder: 3×3 conv (c → c/2) + BN + 1×1 conv (c/2 → 2) + BN.
        total += decoder_params(c_in);

        total
    }
}

// ---------------------------------------------------------------------------
// Per-component parameter formulas
// ---------------------------------------------------------------------------

/// One `InnerGroupedTemporalBlock`: two (depthwise-grouped conv → BN →
/// pointwise conv → BN) stages plus a 1×1 + BN residual projection when the
/// channel count changes. All convs are bias-free.
fn tcn_block_params(c_in: usize, c_out: usize, k: usize, groups: usize) -> usize {
    let grouped1 = c_in * (c_in / groups) * k; // depthwise-grouped, c_in → c_in
    let bn1g = 2 * c_in;
    let pw1 = c_out * c_in; // pointwise 1×1
    let bn1p = 2 * c_out;
    let grouped2 = c_out * (c_out / groups) * k;
    let bn2g = 2 * c_out;
    let pw2 = c_out * c_out;
    let bn2p = 2 * c_out;
    let downsample = if c_in != c_out {
        c_in * c_out + 2 * c_out
    } else {
        0
    };
    grouped1 + bn1g + pw1 + bn1p + grouped2 + bn2g + pw2 + bn2p + downsample
}

/// One `ConvBlock1` / `AsymmetricConvBlock`: three (1, 3) convs **with bias**
/// + BN each, plus a bias-free 1×1 + BN residual projection.
fn conv_block_params(c_in: usize, c_out: usize) -> usize {
    let conv1 = c_out * c_in * 3 + c_out;
    let conv_rest = 2 * (c_out * c_out * 3 + c_out);
    let bns = 3 * 2 * c_out;
    let downsample = c_in * c_out + 2 * c_out;
    conv1 + conv_rest + bns + downsample
}

/// One `AxialAttention` axis: bias-free 1×1 qkv conv (c → 3c), BN over the
/// 3c qkv channels, BN over the `groups` similarity maps, BN over the output.
fn axial_attention_params(c: usize, groups: usize) -> usize {
    let qkv = c * 3 * c;
    let bn_qkv = 2 * (3 * c);
    let bn_similarity = 2 * groups;
    let bn_output = 2 * c;
    qkv + bn_qkv + bn_similarity + bn_output
}

/// Decoder: `Conv2d(c → c/2, 3×3, bias)` + BN + `Conv2d(c/2 → 2, 1×1, bias)`
/// + BN.
fn decoder_params(c: usize) -> usize {
    let mid = c / 2;
    let conv1 = mid * c * 9 + mid;
    let bn1 = 2 * mid;
    let conv2 = 2 * mid + 2;
    let bn2 = 2 * 2;
    conv1 + bn1 + conv2 + bn2
}

// ---------------------------------------------------------------------------
// Tests (pure Rust — run under --no-default-features)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference parameter count verified against the upstream checkpoint
    /// and `torchinfo` (benchmarks/wiflow-std/RESULTS.md, 2026-06-10).
    const REFERENCE_PARAMS: usize = 2_225_042;

    #[test]
    fn default_config_is_valid() {
        WiFlowStdConfig::default()
            .validate()
            .expect("default config must validate");
    }

    #[test]
    fn default_param_count_matches_verified_reference() {
        assert_eq!(WiFlowStdConfig::default().param_count(), REFERENCE_PARAMS);
    }

    #[test]
    fn param_count_is_independent_of_keypoints() {
        // The keypoint count only changes the parameter-free adaptive pool,
        // so 15- and 17-keypoint variants share identical weights.
        let kp17 = WiFlowStdConfig::for_keypoints(17);
        kp17.validate().expect("17-keypoint config must validate");
        assert_eq!(kp17.param_count(), REFERENCE_PARAMS);
    }

    #[test]
    fn per_component_breakdown_matches_hand_calculation() {
        // TCN levels (hand-verified against the reference layer shapes).
        assert_eq!(tcn_block_params(540, 540, 3, 20), 675_000);
        assert_eq!(tcn_block_params(540, 440, 3, 20), 746_180);
        assert_eq!(tcn_block_params(440, 340, 3, 20), 464_780);
        assert_eq!(tcn_block_params(340, 240, 3, 20), 249_380);
        // Conv encoder.
        assert_eq!(conv_block_params(1, 8), 504);
        assert_eq!(conv_block_params(8, 8), 728);
        assert_eq!(conv_block_params(8, 16), 2_224);
        assert_eq!(conv_block_params(16, 32), 8_544);
        assert_eq!(conv_block_params(32, 64), 33_472);
        // Attention + decoder.
        assert_eq!(axial_attention_params(64, 8), 12_816);
        assert_eq!(decoder_params(64), 18_598);
    }

    #[test]
    fn output_shape_default_and_esp32() {
        assert_eq!(WiFlowStdConfig::default().output_shape(4), (4, 15, 2));
        assert_eq!(
            WiFlowStdConfig::for_keypoints(17).output_shape(1),
            (1, 17, 2)
        );
    }

    #[test]
    fn feature_width_default_is_15() {
        // 240 → 120 → 60 → 30 → 15 (four stride-(1,2) blocks).
        assert_eq!(WiFlowStdConfig::default().feature_width(), 15);
    }

    #[test]
    fn tcn_output_channels_default_is_240() {
        assert_eq!(WiFlowStdConfig::default().tcn_output_channels(), 240);
    }

    #[test]
    fn rejects_subcarriers_not_divisible_by_groups() {
        let cfg = WiFlowStdConfig {
            subcarriers: 541,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_dimensions() {
        for cfg in [
            WiFlowStdConfig {
                subcarriers: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                window: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                keypoints: 0,
                ..Default::default()
            },
            WiFlowStdConfig {
                tcn_groups: 0,
                ..Default::default()
            },
        ] {
            assert!(cfg.validate().is_err(), "expected rejection: {cfg:?}");
        }
    }

    #[test]
    fn rejects_empty_or_indivisible_tcn_channels() {
        let empty = WiFlowStdConfig {
            tcn_channels: vec![],
            ..Default::default()
        };
        assert!(empty.validate().is_err());

        let indivisible = WiFlowStdConfig {
            tcn_channels: vec![540, 441],
            ..Default::default()
        };
        assert!(indivisible.validate().is_err());
    }

    #[test]
    fn rejects_bad_conv_channels() {
        let empty = WiFlowStdConfig {
            conv_channels: vec![],
            ..Default::default()
        };
        assert!(empty.validate().is_err());

        let zero = WiFlowStdConfig {
            conv_channels: vec![8, 0, 64],
            ..Default::default()
        };
        assert!(zero.validate().is_err());

        // Odd last channel breaks the c → c/2 decoder split.
        let odd_last = WiFlowStdConfig {
            conv_channels: vec![8, 16, 33],
            attention_groups: 1,
            ..Default::default()
        };
        assert!(odd_last.validate().is_err());
    }

    #[test]
    fn rejects_attention_group_mismatch() {
        let cfg = WiFlowStdConfig {
            attention_groups: 7, // 64 % 7 != 0
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        let zero = WiFlowStdConfig {
            attention_groups: 0,
            ..Default::default()
        };
        assert!(zero.validate().is_err());
    }

    #[test]
    fn rejects_out_of_range_dropout() {
        for d in [1.0, 1.5, -0.1, f64::NAN] {
            let cfg = WiFlowStdConfig {
                dropout: d,
                ..Default::default()
            };
            assert!(cfg.validate().is_err(), "dropout {d} must be rejected");
        }
    }

    #[test]
    fn serde_roundtrip_preserves_config() {
        let cfg = WiFlowStdConfig::for_keypoints(17);
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: WiFlowStdConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cfg);
    }
}
