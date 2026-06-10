//! Personal response vector and session inputs (ADR-250 §6, §9, §10).
//!
//! This is the RuVector layer's data model. The 20-field
//! [`PersonResponseVector`] is the compact adaptive memory updated after each
//! session via [`PersonResponseVector::update`]
//! (`R_{t+1} = update(R_t, stimulus_t, response_t, safety_t)`, ADR-250 §6).

use serde::{Deserialize, Serialize};

use crate::math::clamp_safe;
use crate::stimulus::StimulusParameters;

/// Coarse posture class from RuView passive sensing (ADR-250 §9).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Posture {
    Seated,
    Reclined,
    Supine,
    Standing,
    Unknown,
}

/// Coarse sleep / arousal proxy (ADR-250 §9 item 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SleepState {
    QuietWake,
    Drowsy,
    Asleep,
    Active,
    Unknown,
}

/// Passive, non-camera RuView state for a session (ADR-250 §9, §13
/// `ruview_state`). All scores are `[0,1]` unless noted.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RuViewState {
    /// Breaths per minute.
    pub breathing_rate: f64,
    /// Regularity of the breathing waveform `[0,1]`.
    pub breathing_stability: f64,
    /// Fraction of frames corrupted by motion `[0,1]` (lower is better).
    pub motion_artifact: f64,
    pub posture: Posture,
    /// Fraction of the session the body was still `[0,1]`.
    pub stillness_score: f64,
    /// Fidgeting / restlessness `[0,1]`.
    pub restlessness_score: f64,
    pub sleep_state: SleepState,
    /// Person was present for the session `[0,1]` adherence proxy.
    pub adherence: f64,
    /// Aggregate RuView sensing confidence `[0,1]`.
    pub sensor_confidence: f64,
}

impl RuViewState {
    /// A calm, well-sensed seated baseline — used as a neutral default in
    /// simulation and tests.
    pub fn calm_baseline() -> Self {
        Self {
            breathing_rate: 13.0,
            breathing_stability: 0.85,
            motion_artifact: 0.05,
            posture: Posture::Seated,
            stillness_score: 0.9,
            restlessness_score: 0.1,
            sleep_state: SleepState::QuietWake,
            adherence: 1.0,
            sensor_confidence: 0.9,
        }
    }
}

/// Optional direct EEG entrainment measurement (ADR-250 §13 `eeg_optional`).
/// When absent, the optimizer relies on RF-derived proxies only and must not
/// claim verified neural entrainment (ADR-250 §16 negative consequence 3).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EegMeasurement {
    /// Gamma-band power gain over baseline `[0,1]`.
    pub gamma_power_gain: f64,
    /// Phase-locking value `[0,1]`.
    pub phase_locking_value: f64,
    /// EEG artifact fraction `[0,1]` (lower is better).
    pub artifact_score: f64,
}

/// Participant-reported subjective state (ADR-250 §13 `subjective`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SubjectiveReport {
    pub comfort: f64,
    pub fatigue: f64,
}

impl Default for SubjectiveReport {
    fn default() -> Self {
        Self {
            comfort: 0.85,
            fatigue: 0.2,
        }
    }
}

/// Everything observed during one session, the input to the response update.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SessionObservation {
    pub stimulus: StimulusParameters,
    pub ruview: RuViewState,
    pub eeg: Option<EegMeasurement>,
    pub subjective: SubjectiveReport,
    /// `false` if any safety stop fired this session.
    pub safety_pass: bool,
    /// `true` if an adverse event was recorded (sticky into the vector).
    pub adverse_event: bool,
}

/// The 20-field adaptive personal response vector (ADR-250 §6). Stored as named
/// fields for clarity; [`as_array`](Self::as_array) gives the flat ordered
/// representation used for nearest-neighbor and clustering in RuVector.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PersonResponseVector {
    pub baseline_gamma: f64,
    pub baseline_alpha: f64,
    pub alpha_gamma_ratio: f64,
    pub gamma_power_gain: f64,
    pub phase_locking_value: f64,
    pub breathing_rate: f64,
    pub breathing_stability: f64,
    pub motion_artifact: f64,
    /// Posture encoded as an ordinal `[0,1]` for vector math.
    pub posture_state: f64,
    /// Sleep state encoded as an ordinal `[0,1]`.
    pub sleep_state: f64,
    pub restlessness_score: f64,
    pub stimulus_frequency: f64,
    pub brightness_level: f64,
    pub sound_level: f64,
    pub duty_cycle: f64,
    pub phase_offset: f64,
    pub session_duration: f64,
    pub comfort_score: f64,
    pub adherence_score: f64,
    /// 1.0 once any adverse event has ever occurred for this person (sticky).
    pub adverse_event_flag: f64,
}

impl PersonResponseVector {
    /// Exponential-moving-average weight for session-to-session updates. Small
    /// so a single noisy session cannot dominate (ADR-250 §16 risk
    /// "Over-optimization", mitigation "conservative priors").
    pub const EMA_ALPHA: f64 = 0.3;

    /// Initialize from a baseline reading before any stimulation.
    pub fn baseline(baseline_gamma: f64, baseline_alpha: f64, ruview: &RuViewState) -> Self {
        let ratio = if baseline_gamma > 1e-9 {
            baseline_alpha / baseline_gamma
        } else {
            0.0
        };
        Self {
            baseline_gamma,
            baseline_alpha,
            alpha_gamma_ratio: ratio,
            gamma_power_gain: 0.0,
            phase_locking_value: 0.0,
            breathing_rate: ruview.breathing_rate,
            breathing_stability: ruview.breathing_stability,
            motion_artifact: ruview.motion_artifact,
            posture_state: posture_ordinal(ruview.posture),
            sleep_state: sleep_ordinal(ruview.sleep_state),
            restlessness_score: ruview.restlessness_score,
            stimulus_frequency: 40.0,
            brightness_level: 0.0,
            sound_level: 0.0,
            duty_cycle: 0.0,
            phase_offset: 0.0,
            session_duration: 0.0,
            comfort_score: 0.85,
            adherence_score: ruview.adherence,
            adverse_event_flag: 0.0,
        }
    }

    /// Flat ordered array (ADR-250 §6 field order) for RuVector similarity ops.
    pub fn as_array(&self) -> [f64; 20] {
        [
            self.baseline_gamma,
            self.baseline_alpha,
            self.alpha_gamma_ratio,
            self.gamma_power_gain,
            self.phase_locking_value,
            self.breathing_rate,
            self.breathing_stability,
            self.motion_artifact,
            self.posture_state,
            self.sleep_state,
            self.restlessness_score,
            self.stimulus_frequency,
            self.brightness_level,
            self.sound_level,
            self.duty_cycle,
            self.phase_offset,
            self.session_duration,
            self.comfort_score,
            self.adherence_score,
            self.adverse_event_flag,
        ]
    }

    /// `R_{t+1} = update(R_t, stimulus_t, response_t, safety_t)` (ADR-250 §6).
    ///
    /// EMA-blends the continuous response fields toward the latest observation;
    /// the adverse-event flag is *sticky* (monotonic 0→1) so a person's safety
    /// history can never be smoothed away.
    pub fn update(&mut self, obs: &SessionObservation) {
        let a = Self::EMA_ALPHA;
        let blend = |old: f64, new: f64| old + a * (new - old);

        if let Some(eeg) = obs.eeg {
            self.gamma_power_gain = blend(self.gamma_power_gain, eeg.gamma_power_gain);
            self.phase_locking_value = blend(self.phase_locking_value, eeg.phase_locking_value);
        }
        self.breathing_rate = blend(self.breathing_rate, obs.ruview.breathing_rate);
        self.breathing_stability =
            blend(self.breathing_stability, obs.ruview.breathing_stability);
        self.motion_artifact = blend(self.motion_artifact, obs.ruview.motion_artifact);
        self.posture_state = posture_ordinal(obs.ruview.posture);
        self.sleep_state = sleep_ordinal(obs.ruview.sleep_state);
        self.restlessness_score = blend(self.restlessness_score, obs.ruview.restlessness_score);
        self.stimulus_frequency = obs.stimulus.frequency_hz;
        self.brightness_level = obs.stimulus.brightness_level;
        self.sound_level = obs.stimulus.volume_level;
        self.duty_cycle = duty_ordinal(obs.stimulus.duty_cycle);
        self.phase_offset = obs.stimulus.phase_offset_ms;
        self.session_duration = obs.stimulus.duration_minutes;
        self.comfort_score = blend(self.comfort_score, obs.subjective.comfort);
        self.adherence_score = blend(self.adherence_score, obs.ruview.adherence);
        if obs.adverse_event {
            self.adverse_event_flag = 1.0; // sticky
        }
        // Keep all `[0,1]` fields well-formed regardless of upstream noise.
        self.clamp_unit_fields();
    }

    fn clamp_unit_fields(&mut self) {
        for f in [
            &mut self.gamma_power_gain,
            &mut self.phase_locking_value,
            &mut self.breathing_stability,
            &mut self.motion_artifact,
            &mut self.restlessness_score,
            &mut self.comfort_score,
            &mut self.adherence_score,
        ] {
            *f = clamp_safe(*f, 0.0, 1.0);
        }
    }
}

fn posture_ordinal(p: Posture) -> f64 {
    match p {
        Posture::Standing => 0.0,
        Posture::Seated => 0.33,
        Posture::Reclined => 0.66,
        Posture::Supine => 1.0,
        Posture::Unknown => 0.5,
    }
}

fn sleep_ordinal(s: SleepState) -> f64 {
    match s {
        SleepState::Active => 0.0,
        SleepState::QuietWake => 0.33,
        SleepState::Drowsy => 0.66,
        SleepState::Asleep => 1.0,
        SleepState::Unknown => 0.5,
    }
}

fn duty_ordinal(d: crate::stimulus::DutyCycle) -> f64 {
    match d {
        crate::stimulus::DutyCycle::Continuous => 0.0,
        crate::stimulus::DutyCycle::Ramped => 0.5,
        crate::stimulus::DutyCycle::Pulsed => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::StimulusParameters;

    fn obs_with_adverse(adverse: bool) -> SessionObservation {
        SessionObservation {
            stimulus: StimulusParameters::prior(),
            ruview: RuViewState::calm_baseline(),
            eeg: Some(EegMeasurement {
                gamma_power_gain: 0.4,
                phase_locking_value: 0.6,
                artifact_score: 0.05,
            }),
            subjective: SubjectiveReport::default(),
            safety_pass: !adverse,
            adverse_event: adverse,
        }
    }

    #[test]
    fn vector_has_twenty_fields() {
        let v = PersonResponseVector::baseline(0.2, 0.5, &RuViewState::calm_baseline());
        assert_eq!(v.as_array().len(), 20);
    }

    #[test]
    fn update_moves_gamma_toward_observation() {
        let mut v = PersonResponseVector::baseline(0.2, 0.5, &RuViewState::calm_baseline());
        let before = v.gamma_power_gain;
        v.update(&obs_with_adverse(false));
        assert!(v.gamma_power_gain > before);
        assert!(v.gamma_power_gain < 0.4); // EMA, not a jump to the target
    }

    #[test]
    fn adverse_flag_is_sticky() {
        let mut v = PersonResponseVector::baseline(0.2, 0.5, &RuViewState::calm_baseline());
        v.update(&obs_with_adverse(true));
        assert_eq!(v.adverse_event_flag, 1.0);
        // A subsequent clean session must NOT clear the flag.
        v.update(&obs_with_adverse(false));
        assert_eq!(v.adverse_event_flag, 1.0);
    }

    #[test]
    fn unit_fields_stay_bounded_under_noise() {
        let mut v = PersonResponseVector::baseline(0.2, 0.5, &RuViewState::calm_baseline());
        let mut obs = obs_with_adverse(false);
        obs.eeg = Some(EegMeasurement {
            gamma_power_gain: 99.0,
            phase_locking_value: -5.0,
            artifact_score: 0.0,
        });
        v.update(&obs);
        assert!(v.gamma_power_gain <= 1.0);
        assert!(v.phase_locking_value >= 0.0);
    }
}
