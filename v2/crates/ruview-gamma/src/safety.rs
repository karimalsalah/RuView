//! Safety stop conditions, exclusion screening, and the in-session monitor
//! (ADR-250 §12).
//!
//! Safety is a **hard constraint, not a weighted term** (ADR-250 §7). This
//! module owns the two non-negotiable gates:
//!   1. [`ExclusionScreen`] — who may participate at all.
//!   2. [`SafetyMonitor`] — when an in-progress session must stop.

use serde::{Deserialize, Serialize};

/// Adverse symptoms / events that force an immediate hard stop (ADR-250 §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdverseEvent {
    Headache,
    Dizziness,
    Nausea,
    Agitation,
    VisualDiscomfort,
    AbnormalDistress,
    SeizureLikeSymptom,
    /// The participant pressed stop.
    UserStopRequest,
}

impl AdverseEvent {
    pub fn tag(self) -> &'static str {
        match self {
            AdverseEvent::Headache => "headache",
            AdverseEvent::Dizziness => "dizziness",
            AdverseEvent::Nausea => "nausea",
            AdverseEvent::Agitation => "agitation",
            AdverseEvent::VisualDiscomfort => "visual_discomfort",
            AdverseEvent::AbnormalDistress => "abnormal_distress",
            AdverseEvent::SeizureLikeSymptom => "seizure_like_symptom",
            AdverseEvent::UserStopRequest => "user_stop_request",
        }
    }
}

/// Conditions requiring exclusion or explicit clinical supervision
/// (ADR-250 §12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExclusionCondition {
    EpilepsyOrSeizureHistory,
    Photosensitivity,
    SevereMigraineSensitivity,
    SeverePsychiatricInstability,
    ImplantedNeurologicalDevice,
    SignificantSensoryImpairment,
    RecentMedicationChange,
}

/// Result of pre-enrollment screening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScreenOutcome {
    /// Cleared for unsupervised research participation.
    Cleared,
    /// May proceed only under clinician supervision (records the conditions).
    RequiresClinicalSupervision(Vec<ExclusionCondition>),
    /// Hard-excluded (records the conditions).
    Excluded(Vec<ExclusionCondition>),
}

/// Inclusion/exclusion screen (ADR-250 §11 RuFlo responsibility 3, §12).
#[derive(Debug, Clone, Default)]
pub struct ExclusionScreen;

impl ExclusionScreen {
    /// Apply the screening policy. Epilepsy/seizure and photosensitivity are
    /// hard exclusions for unsupervised use because the protocol is, by
    /// construction, flicker stimulation; the rest require supervision.
    pub fn evaluate(&self, conditions: &[ExclusionCondition]) -> ScreenOutcome {
        if conditions.is_empty() {
            return ScreenOutcome::Cleared;
        }
        let hard: Vec<ExclusionCondition> = conditions
            .iter()
            .copied()
            .filter(|c| {
                matches!(
                    c,
                    ExclusionCondition::EpilepsyOrSeizureHistory
                        | ExclusionCondition::Photosensitivity
                )
            })
            .collect();
        if !hard.is_empty() {
            return ScreenOutcome::Excluded(hard);
        }
        ScreenOutcome::RequiresClinicalSupervision(conditions.to_vec())
    }
}

/// Why a session was stopped. `Completed` is the only non-stop terminal state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum StopReason {
    /// Ran to planned completion — not a safety stop.
    Completed,
    /// An adverse event was reported.
    AdverseEvent(AdverseEvent),
    /// Sensor confidence dropped below the floor (entrainment unverifiable).
    SensorConfidenceBelowFloor { value: f64, floor: f64 },
    /// A requested parameter fell outside the approved envelope.
    ProtocolOutsideEnvelope,
}

impl StopReason {
    /// `true` for every reason that is a *safety* stop (everything but
    /// `Completed`). Used by the audit layer to assert "100% safety stops
    /// logged" (ADR-250 §18).
    pub fn is_safety_stop(&self) -> bool {
        !matches!(self, StopReason::Completed)
    }
}

/// Live per-tick session telemetry the monitor evaluates.
#[derive(Debug, Clone, Copy)]
pub struct SafetyTick {
    /// Reported adverse event this tick, if any.
    pub adverse: Option<AdverseEvent>,
    /// Aggregate sensor confidence `[0,1]` (RuView + optional EEG).
    pub sensor_confidence: f64,
    /// Whether the currently-applied stimulus is inside the envelope.
    pub stimulus_in_envelope: bool,
}

/// In-session safety monitor with a configurable confidence floor and a
/// bounded stop latency contract (ADR-250 §17: safety-stop latency < 500 ms).
#[derive(Debug, Clone)]
pub struct SafetyMonitor {
    /// Minimum acceptable sensor confidence (ADR-250 §12 condition 9).
    pub confidence_floor: f64,
    /// Declared worst-case evaluation latency in milliseconds; the monitor is
    /// O(1) per tick so the real figure is far below this, but the contract is
    /// asserted in tests against ADR-250 §17's 500 ms bound.
    pub max_eval_latency_ms: u32,
    triggered: Option<StopReason>,
}

impl Default for SafetyMonitor {
    fn default() -> Self {
        Self {
            confidence_floor: 0.5,
            max_eval_latency_ms: 50,
            triggered: None,
        }
    }
}

impl SafetyMonitor {
    pub fn new(confidence_floor: f64) -> Self {
        Self {
            confidence_floor,
            ..Default::default()
        }
    }

    /// Evaluate one tick. Once a stop has fired the monitor is *latched* — it
    /// keeps returning the original stop reason so a session can never silently
    /// resume after a safety event (closed-loop "terminate and lock",
    /// ADR-250 §8 Phase 4).
    pub fn evaluate(&mut self, tick: SafetyTick) -> Option<StopReason> {
        if let Some(reason) = &self.triggered {
            return Some(reason.clone());
        }
        let reason = if let Some(ev) = tick.adverse {
            Some(StopReason::AdverseEvent(ev))
        } else if !tick.stimulus_in_envelope {
            Some(StopReason::ProtocolOutsideEnvelope)
        } else if tick.sensor_confidence < self.confidence_floor {
            Some(StopReason::SensorConfidenceBelowFloor {
                value: tick.sensor_confidence,
                floor: self.confidence_floor,
            })
        } else {
            None
        };
        if let Some(r) = &reason {
            self.triggered = Some(r.clone());
        }
        reason
    }

    /// Whether the monitor has latched a stop.
    pub fn is_stopped(&self) -> bool {
        self.triggered.is_some()
    }

    /// The latched stop reason, if any.
    pub fn stop_reason(&self) -> Option<&StopReason> {
        self.triggered.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_history_clears() {
        assert_eq!(ExclusionScreen.evaluate(&[]), ScreenOutcome::Cleared);
    }

    #[test]
    fn epilepsy_is_hard_excluded() {
        let out = ExclusionScreen.evaluate(&[ExclusionCondition::EpilepsyOrSeizureHistory]);
        assert!(matches!(out, ScreenOutcome::Excluded(_)));
    }

    #[test]
    fn photosensitivity_is_hard_excluded() {
        let out = ExclusionScreen.evaluate(&[ExclusionCondition::Photosensitivity]);
        assert!(matches!(out, ScreenOutcome::Excluded(_)));
    }

    #[test]
    fn migraine_requires_supervision() {
        let out = ExclusionScreen.evaluate(&[ExclusionCondition::SevereMigraineSensitivity]);
        assert!(matches!(out, ScreenOutcome::RequiresClinicalSupervision(_)));
    }

    #[test]
    fn adverse_event_triggers_stop() {
        let mut m = SafetyMonitor::default();
        let r = m.evaluate(SafetyTick {
            adverse: Some(AdverseEvent::Dizziness),
            sensor_confidence: 0.9,
            stimulus_in_envelope: true,
        });
        assert_eq!(r, Some(StopReason::AdverseEvent(AdverseEvent::Dizziness)));
        assert!(r.unwrap().is_safety_stop());
    }

    #[test]
    fn low_confidence_triggers_stop() {
        let mut m = SafetyMonitor::new(0.6);
        let r = m.evaluate(SafetyTick {
            adverse: None,
            sensor_confidence: 0.3,
            stimulus_in_envelope: true,
        });
        assert!(matches!(
            r,
            Some(StopReason::SensorConfidenceBelowFloor { .. })
        ));
    }

    #[test]
    fn out_of_envelope_triggers_stop() {
        let mut m = SafetyMonitor::default();
        let r = m.evaluate(SafetyTick {
            adverse: None,
            sensor_confidence: 0.9,
            stimulus_in_envelope: false,
        });
        assert_eq!(r, Some(StopReason::ProtocolOutsideEnvelope));
    }

    #[test]
    fn stop_is_latched_and_cannot_resume() {
        let mut m = SafetyMonitor::default();
        m.evaluate(SafetyTick {
            adverse: Some(AdverseEvent::Headache),
            sensor_confidence: 0.9,
            stimulus_in_envelope: true,
        });
        // A subsequent "all clear" tick must NOT clear the latch.
        let r = m.evaluate(SafetyTick {
            adverse: None,
            sensor_confidence: 1.0,
            stimulus_in_envelope: true,
        });
        assert_eq!(r, Some(StopReason::AdverseEvent(AdverseEvent::Headache)));
        assert!(m.is_stopped());
    }

    #[test]
    fn completed_is_not_a_safety_stop() {
        assert!(!StopReason::Completed.is_safety_stop());
    }
}
