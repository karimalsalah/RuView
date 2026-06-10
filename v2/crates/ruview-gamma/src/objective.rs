//! The safe-entrainment objective (ADR-250 §7).
//!
//! `score = w1·gamma + w2·phase_locking + w3·breathing_stability + w4·adherence
//!        + w5·comfort − w6·motion_artifact − w7·adverse_risk − w8·overstim`
//!
//! Safety is **not** a soft term here: it is a hard gate applied *before*
//! scoring (`SafetyEnvelope`, `SafetyMonitor`). The `adverse_event_risk` and
//! `overstimulation_penalty` terms only shape preference *within* the already
//! safe region.

use serde::{Deserialize, Serialize};

use crate::response::{EegMeasurement, RuViewState, SubjectiveReport};
use crate::stimulus::{SafetyEnvelope, StimulusParameters};

/// Objective weights (ADR-250 §7 default weighting).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub gamma_gain: f64,
    pub phase_locking: f64,
    pub breathing_stability: f64,
    pub adherence: f64,
    pub comfort: f64,
    pub motion_artifact: f64,
    pub adverse_event_risk: f64,
    pub overstimulation: f64,
}

impl Default for ObjectiveWeights {
    /// ADR-250 §7 default weighting table.
    fn default() -> Self {
        Self {
            gamma_gain: 0.30,
            phase_locking: 0.25,
            breathing_stability: 0.10,
            adherence: 0.10,
            comfort: 0.15,
            motion_artifact: 0.05,
            adverse_event_risk: 0.20,
            overstimulation: 0.10,
        }
    }
}

/// Inputs to the score, gathered from one session's observations.
#[derive(Debug, Clone, Copy)]
pub struct ScoreInputs<'a> {
    pub stimulus: &'a StimulusParameters,
    pub ruview: &'a RuViewState,
    pub eeg: Option<&'a EegMeasurement>,
    pub subjective: &'a SubjectiveReport,
    /// Per-session adverse-event risk estimate `[0,1]` (model-provided).
    pub adverse_event_risk: f64,
}

/// The safe-entrainment objective.
#[derive(Debug, Clone, Copy)]
pub struct SafeEntrainmentObjective {
    pub weights: ObjectiveWeights,
    pub envelope: SafetyEnvelope,
}

impl SafeEntrainmentObjective {
    pub fn new(weights: ObjectiveWeights, envelope: SafetyEnvelope) -> Self {
        Self { weights, envelope }
    }

    /// Score a session in `[~−1, ~1]`. When EEG is absent, the gamma and
    /// phase-locking terms fall back to RF-derived proxies (stillness and
    /// breathing stability) at reduced weight — never claiming verified neural
    /// entrainment (ADR-250 §16 consequence 3).
    pub fn score(&self, inp: &ScoreInputs) -> f64 {
        let w = &self.weights;

        let (gamma, plv) = match inp.eeg {
            Some(e) => (e.gamma_power_gain, e.phase_locking_value),
            // RF-only proxy: down-weighted, derived from calm/still physiology.
            None => (
                0.5 * inp.ruview.stillness_score,
                0.5 * inp.ruview.breathing_stability,
            ),
        };

        let overstim = self.overstimulation_penalty(inp.stimulus);

        w.gamma_gain * gamma
            + w.phase_locking * plv
            + w.breathing_stability * inp.ruview.breathing_stability
            + w.adherence * inp.ruview.adherence
            + w.comfort * inp.subjective.comfort
            - w.motion_artifact * inp.ruview.motion_artifact
            - w.adverse_event_risk * inp.adverse_event_risk
            - w.overstimulation * overstim
    }

    /// Overstimulation penalty `[0,1]`: how close intensity sits to the caps,
    /// plus a duration term. Penalizes pushing brightness/volume toward the
    /// envelope edges and running long sessions before tolerance is shown.
    pub fn overstimulation_penalty(&self, s: &StimulusParameters) -> f64 {
        let b = if self.envelope.brightness_cap > 0.0 {
            s.brightness_level / self.envelope.brightness_cap
        } else {
            0.0
        };
        let v = if self.envelope.volume_cap > 0.0 {
            s.volume_level / self.envelope.volume_cap
        } else {
            0.0
        };
        let d = if self.envelope.max_duration_minutes > 0.0 {
            s.duration_minutes / self.envelope.max_duration_minutes
        } else {
            0.0
        };
        // Mean of the three normalized loads, clamped to [0,1].
        ((b + v + d) / 3.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::response::SubjectiveReport;

    fn objective() -> SafeEntrainmentObjective {
        SafeEntrainmentObjective::new(ObjectiveWeights::default(), SafetyEnvelope::conservative())
    }

    #[test]
    fn weights_sum_to_documented_totals() {
        let w = ObjectiveWeights::default();
        // Positive entrainment+experience terms sum to 0.90 (ADR-250 §7).
        let pos = w.gamma_gain + w.phase_locking + w.breathing_stability + w.adherence + w.comfort;
        assert!((pos - 0.90).abs() < 1e-9);
    }

    #[test]
    fn strong_entrainment_scores_higher_than_weak() {
        let obj = objective();
        let stim = StimulusParameters::prior();
        let ruview = RuViewState::calm_baseline();
        let subj = SubjectiveReport::default();
        let strong = EegMeasurement {
            gamma_power_gain: 0.8,
            phase_locking_value: 0.8,
            artifact_score: 0.02,
        };
        let weak = EegMeasurement {
            gamma_power_gain: 0.1,
            phase_locking_value: 0.1,
            artifact_score: 0.02,
        };
        let s_strong = obj.score(&ScoreInputs {
            stimulus: &stim,
            ruview: &ruview,
            eeg: Some(&strong),
            subjective: &subj,
            adverse_event_risk: 0.0,
        });
        let s_weak = obj.score(&ScoreInputs {
            stimulus: &stim,
            ruview: &ruview,
            eeg: Some(&weak),
            subjective: &subj,
            adverse_event_risk: 0.0,
        });
        assert!(s_strong > s_weak);
    }

    #[test]
    fn adverse_risk_reduces_score() {
        let obj = objective();
        let stim = StimulusParameters::prior();
        let ruview = RuViewState::calm_baseline();
        let subj = SubjectiveReport::default();
        let eeg = EegMeasurement {
            gamma_power_gain: 0.5,
            phase_locking_value: 0.5,
            artifact_score: 0.02,
        };
        let safe = obj.score(&ScoreInputs {
            stimulus: &stim,
            ruview: &ruview,
            eeg: Some(&eeg),
            subjective: &subj,
            adverse_event_risk: 0.0,
        });
        let risky = obj.score(&ScoreInputs {
            stimulus: &stim,
            ruview: &ruview,
            eeg: Some(&eeg),
            subjective: &subj,
            adverse_event_risk: 0.9,
        });
        assert!(risky < safe);
    }

    #[test]
    fn overstimulation_penalty_grows_with_intensity() {
        let obj = objective();
        let mut low = StimulusParameters::prior();
        low.brightness_level = 0.05;
        low.volume_level = 0.05;
        low.duration_minutes = 5.0;
        let mut high = StimulusParameters::prior();
        high.brightness_level = 0.40;
        high.volume_level = 0.40;
        high.duration_minutes = 15.0;
        assert!(obj.overstimulation_penalty(&high) > obj.overstimulation_penalty(&low));
    }

    #[test]
    fn no_eeg_falls_back_to_rf_proxy() {
        let obj = objective();
        let stim = StimulusParameters::prior();
        let ruview = RuViewState::calm_baseline();
        let subj = SubjectiveReport::default();
        // Should produce a finite, sane score without EEG.
        let s = obj.score(&ScoreInputs {
            stimulus: &stim,
            ruview: &ruview,
            eeg: None,
            subjective: &subj,
            adverse_event_risk: 0.0,
        });
        assert!(s.is_finite());
    }
}
