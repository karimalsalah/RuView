//! Synthetic response simulator — Milestone 1 (ADR-250 §21).
//!
//! `frequency_response_curve(person, state, stimulus)` — a deterministic stand-in
//! for real EEG + RuView measurements so the optimizer, safety bounds, and
//! RuVector update logic can be exercised and replayed bit-exactly **before any
//! hardware or human exposure**.
//!
//! Determinism contract (same discipline as `nvsim`): the response for a given
//! `(global_seed, person_id, session_index, stimulus)` is byte-identical across
//! runs and machines. Noise is drawn from a ChaCha20 stream seeded from a
//! SHA-256 of those inputs — no OS entropy, no wall clock.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::response::{EegMeasurement, RuViewState, SleepState};
use crate::stimulus::StimulusParameters;

/// A person's hidden response physiology. Real people are not 40 Hz-identical;
/// each has a latent best frequency the optimizer must *discover* (ADR-250 §1,
/// the 2025 PLOS One 36–44 Hz finding).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatentPerson {
    /// The individual's true peak entrainment frequency in Hz.
    pub peak_hz: f64,
    /// Tuning-curve width in Hz (smaller = sharper, harder to find).
    pub width_hz: f64,
    /// Maximum achievable gamma gain at the peak `[0,1]`.
    pub max_gain: f64,
    /// How quickly comfort falls as intensity rises `[0,1]`.
    pub intensity_sensitivity: f64,
    /// Intrinsic measurement noise level `[0,1]`.
    pub noise: f64,
}

impl LatentPerson {
    /// Derive a stable latent person from a pseudonymous id. Deterministic: the
    /// same id always yields the same physiology, so multi-session studies are
    /// reproducible.
    pub fn from_id(person_id: &str) -> Self {
        let h = stable_hash(&[b"latent-person", person_id.as_bytes()]);
        // Map hash bytes to parameter ranges.
        let u = |i: usize| (h[i] as f64) / 255.0;
        Self {
            peak_hz: 37.0 + u(0) * 6.0,             // 37..43 Hz
            width_hz: 1.5 + u(1) * 2.5,             // 1.5..4.0 Hz
            max_gain: 0.45 + u(2) * 0.45,           // 0.45..0.90
            intensity_sensitivity: 0.2 + u(3) * 0.6, // 0.2..0.8
            noise: 0.02 + u(4) * 0.08,              // 0.02..0.10
        }
    }
}

/// One simulated session outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulatedResponse {
    /// Synthetic EEG entrainment measurement.
    pub eeg: EegMeasurement,
    /// Updated RuView state for this session (motion/comfort feedback).
    pub ruview: RuViewState,
    /// Subjective comfort `[0,1]`.
    pub comfort: f64,
    /// Whether an adverse event occurred (rises with overstimulation).
    pub adverse_event: bool,
}

/// Deterministic response simulator.
#[derive(Debug, Clone)]
pub struct ResponseSimulator {
    seed: u64,
}

impl ResponseSimulator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Simulate a session. `session_index` makes repeated identical protocols
    /// produce independent (but reproducible) noise draws.
    pub fn simulate(
        &self,
        person: &LatentPerson,
        state: &RuViewState,
        stimulus: &StimulusParameters,
        session_index: u64,
    ) -> SimulatedResponse {
        let mut rng = self.session_rng(person, stimulus, session_index);

        // --- Frequency tuning curve: Gaussian bump around the latent peak. ---
        let df = stimulus.frequency_hz - person.peak_hz;
        let tuning = (-0.5 * (df / person.width_hz).powi(2)).exp();

        // --- State modulation: calm, still, awake-but-quiet entrains best. ---
        let state_factor = state_modulation(state);

        // --- Intensity helps gamma up to a point, then risks overstimulation. ---
        let intensity = 0.5 * (stimulus.brightness_level + stimulus.volume_level);
        // Mild positive contribution from intensity, saturating.
        let intensity_gain = 0.6 + 0.4 * (1.0 - (-3.0 * intensity).exp());

        let mut noise = || -> f64 { (rng.gen::<f64>() - 0.5) * 2.0 * person.noise };

        let gamma_power_gain =
            clamp01(person.max_gain * tuning * state_factor * intensity_gain + noise());
        // Phase locking tracks tuning but is less intensity-dependent.
        let phase_locking_value = clamp01(0.9 * tuning * state_factor + noise());
        // Artifact rises with motion and intensity.
        let artifact_score = clamp01(state.motion_artifact + 0.1 * intensity + noise().abs());

        // --- Comfort: high intensity and long duration erode comfort. ---
        let dur_load = (stimulus.duration_minutes / 15.0).clamp(0.0, 1.0);
        let comfort = clamp01(
            0.95 - person.intensity_sensitivity * (0.7 * intensity + 0.3 * dur_load) + noise(),
        );

        // --- Adverse event: rare, and only when truly overstimulated. ---
        let overstim = intensity * dur_load;
        let adverse_event = overstim > 0.85 && rng.gen::<f64>() < (overstim - 0.85) * 2.0;

        // Feedback into RuView state: discomfort raises motion/restlessness.
        let mut ruview = *state;
        ruview.motion_artifact = clamp01(state.motion_artifact + (1.0 - comfort) * 0.1);
        ruview.restlessness_score = clamp01(state.restlessness_score + (1.0 - comfort) * 0.15);

        SimulatedResponse {
            eeg: EegMeasurement {
                gamma_power_gain,
                phase_locking_value,
                artifact_score,
            },
            ruview,
            comfort,
            adverse_event,
        }
    }

    /// Per-session ChaCha20 stream seeded from all inputs — reproducible noise.
    fn session_rng(
        &self,
        person: &LatentPerson,
        stimulus: &StimulusParameters,
        session_index: u64,
    ) -> ChaCha20Rng {
        // Quantize stimulus to 0.1 Hz / 0.01 intensity so floating noise in the
        // last bits cannot fork the stream; matches the ±0.1 Hz control spec.
        let q_freq = (stimulus.frequency_hz * 10.0).round() as i64;
        let q_bright = (stimulus.brightness_level * 100.0).round() as i64;
        let q_vol = (stimulus.volume_level * 100.0).round() as i64;
        let q_peak = (person.peak_hz * 1000.0).round() as i64;
        let h = stable_hash(&[
            b"session-rng",
            &self.seed.to_le_bytes(),
            &session_index.to_le_bytes(),
            &q_freq.to_le_bytes(),
            &q_bright.to_le_bytes(),
            &q_vol.to_le_bytes(),
            &q_peak.to_le_bytes(),
        ]);
        let mut seed32 = [0u8; 32];
        seed32.copy_from_slice(&h);
        ChaCha20Rng::from_seed(seed32)
    }
}

fn state_modulation(state: &RuViewState) -> f64 {
    let sleep = match state.sleep_state {
        SleepState::QuietWake => 1.0,
        SleepState::Drowsy => 0.85,
        SleepState::Asleep => 0.6,
        SleepState::Active => 0.7,
        SleepState::Unknown => 0.8,
    };
    // Stillness and breathing stability help; restlessness hurts.
    let calm = 0.5 + 0.3 * state.stillness_score + 0.2 * state.breathing_stability
        - 0.2 * state.restlessness_score;
    (sleep * calm).clamp(0.0, 1.0)
}

#[inline]
fn clamp01(v: f64) -> f64 {
    if v.is_nan() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

/// SHA-256 over concatenated byte chunks → 32 bytes. Deterministic and
/// portable; the witness foundation for the whole crate.
pub(crate) fn stable_hash(chunks: &[&[u8]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for c in chunks {
        hasher.update((c.len() as u64).to_le_bytes());
        hasher.update(c);
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::StimulusParameters;

    #[test]
    fn same_inputs_produce_identical_output() {
        let sim = ResponseSimulator::new(42);
        let person = LatentPerson::from_id("subject-A");
        let state = RuViewState::calm_baseline();
        let stim = StimulusParameters::prior();
        let a = sim.simulate(&person, &state, &stim, 0);
        let b = sim.simulate(&person, &state, &stim, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn different_session_index_changes_noise() {
        let sim = ResponseSimulator::new(42);
        let person = LatentPerson::from_id("subject-A");
        let state = RuViewState::calm_baseline();
        let stim = StimulusParameters::prior();
        let a = sim.simulate(&person, &state, &stim, 0);
        let b = sim.simulate(&person, &state, &stim, 1);
        assert_ne!(a.eeg.gamma_power_gain, b.eeg.gamma_power_gain);
    }

    #[test]
    fn latent_person_is_stable_per_id() {
        assert_eq!(
            LatentPerson::from_id("subject-A"),
            LatentPerson::from_id("subject-A")
        );
        assert_ne!(
            LatentPerson::from_id("subject-A").peak_hz,
            LatentPerson::from_id("subject-Z").peak_hz
        );
    }

    #[test]
    fn peak_frequency_entrains_better_than_band_edge() {
        let sim = ResponseSimulator::new(7);
        let person = LatentPerson::from_id("subject-B");
        let state = RuViewState::calm_baseline();

        let mut at_peak = StimulusParameters::prior();
        at_peak.frequency_hz = (person.peak_hz * 10.0).round() / 10.0;
        let mut at_edge = StimulusParameters::prior();
        at_edge.frequency_hz = 36.0;

        let r_peak = sim.simulate(&person, &state, &at_peak, 0);
        let r_edge = sim.simulate(&person, &state, &at_edge, 0);
        // Only assert when the latent peak is genuinely away from the edge.
        if (person.peak_hz - 36.0).abs() > 1.0 {
            assert!(r_peak.eeg.gamma_power_gain > r_edge.eeg.gamma_power_gain);
        }
    }

    #[test]
    fn calm_state_entrains_better_than_restless() {
        let sim = ResponseSimulator::new(3);
        let person = LatentPerson::from_id("subject-C");
        let stim = StimulusParameters::prior();

        let calm = RuViewState::calm_baseline();
        let mut restless = RuViewState::calm_baseline();
        restless.restlessness_score = 0.9;
        restless.stillness_score = 0.2;
        restless.sleep_state = SleepState::Active;

        let r_calm = sim.simulate(&person, &calm, &stim, 0);
        let r_restless = sim.simulate(&person, &restless, &stim, 0);
        assert!(r_calm.eeg.gamma_power_gain > r_restless.eeg.gamma_power_gain);
    }
}
