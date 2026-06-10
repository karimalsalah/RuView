//! Session record and the reproducible session witness (ADR-250 §11, §13).
//!
//! `session_hash = hash(protocol_version, model_version, device_version,
//! stimulus_parameters, sensor_summary, response_summary, safety_events)`.
//!
//! The hash is computed over a **canonical** serialization (fixed field order,
//! quantized floats) so the same session content always yields the same digest
//! across machines — the RuFlo reproducibility trail. The [`SessionId`] is
//! *derived from* the hash, so identifiers are themselves reproducible (no
//! random UUIDs that would break replay).

use serde::{Deserialize, Serialize};

use crate::response::{EegMeasurement, RuViewState, SubjectiveReport};
use crate::safety::StopReason;
use crate::simulator::stable_hash;
use crate::stimulus::StimulusParameters;

/// Version triple identifying the exact software/hardware that ran a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionTriple {
    pub protocol_version: String,
    pub model_version: String,
    pub device_version: String,
}

impl Default for VersionTriple {
    fn default() -> Self {
        Self {
            protocol_version: "adr-250-v0.1".to_string(),
            model_version: "ruvector-gamma-v0.1".to_string(),
            device_version: "sim-harness-v0.1".to_string(),
        }
    }
}

/// Per-session outcome summary (ADR-250 §13 `outcome`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    pub entrainment_score: f64,
    pub safety_pass: bool,
    pub recommended_next_frequency_hz: f64,
}

/// A complete, hashable session record (ADR-250 §13).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Derived from the content hash (see [`SessionRecord::finalize`]).
    pub session_id: SessionId,
    /// Pseudonymous participant id.
    pub person_id: String,
    pub versions: VersionTriple,
    /// Caller-supplied epoch milliseconds (kept explicit for determinism — no
    /// wall-clock reads inside the crate).
    pub timestamp_ms: u64,
    pub stimulus: StimulusParameters,
    pub ruview_state: RuViewState,
    pub eeg_optional: Option<EegMeasurement>,
    pub subjective: SubjectiveReport,
    pub outcome: Outcome,
    /// Every safety event raised during the session (ADR-250 §18: 100% logged).
    pub safety_events: Vec<StopReason>,
    /// The session witness (hex SHA-256).
    pub session_hash: String,
}

/// A reproducible session identifier: the first 16 hex chars of the witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionId(pub String);

/// Builder that gathers session inputs and finalizes them into an immutable,
/// witnessed [`SessionRecord`].
#[derive(Debug, Clone)]
pub struct SessionBuilder {
    person_id: String,
    versions: VersionTriple,
    timestamp_ms: u64,
    stimulus: StimulusParameters,
    ruview: RuViewState,
    eeg: Option<EegMeasurement>,
    subjective: SubjectiveReport,
    outcome: Outcome,
    safety_events: Vec<StopReason>,
}

impl SessionBuilder {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        person_id: impl Into<String>,
        versions: VersionTriple,
        timestamp_ms: u64,
        stimulus: StimulusParameters,
        ruview: RuViewState,
        subjective: SubjectiveReport,
        outcome: Outcome,
    ) -> Self {
        Self {
            person_id: person_id.into(),
            versions,
            timestamp_ms,
            stimulus,
            ruview,
            eeg: None,
            subjective,
            outcome,
            safety_events: Vec::new(),
        }
    }

    pub fn with_eeg(mut self, eeg: EegMeasurement) -> Self {
        self.eeg = Some(eeg);
        self
    }

    pub fn with_safety_events(mut self, events: Vec<StopReason>) -> Self {
        self.safety_events = events;
        self
    }

    /// Compute the canonical witness and freeze the record.
    pub fn finalize(self) -> SessionRecord {
        let canon = self.canonical_bytes();
        let digest = stable_hash(&[&canon]);
        let hex = hex_encode(&digest);
        let session_id = SessionId(hex[..16].to_string());
        SessionRecord {
            session_id,
            person_id: self.person_id,
            versions: self.versions,
            timestamp_ms: self.timestamp_ms,
            stimulus: self.stimulus,
            ruview_state: self.ruview,
            eeg_optional: self.eeg,
            subjective: self.subjective,
            outcome: self.outcome,
            safety_events: self.safety_events,
            session_hash: hex,
        }
    }

    /// Canonical byte serialization in the ADR-250 §11 hash field order.
    /// Floats are quantized so last-bit jitter never forks the witness.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut s = String::new();
        s.push_str(&self.versions.protocol_version);
        s.push('|');
        s.push_str(&self.versions.model_version);
        s.push('|');
        s.push_str(&self.versions.device_version);
        s.push('|');
        // stimulus_parameters
        push_q(&mut s, self.stimulus.frequency_hz, 10.0);
        s.push_str(self.stimulus.modality.tag());
        push_q(&mut s, self.stimulus.brightness_level, 100.0);
        push_q(&mut s, self.stimulus.volume_level, 100.0);
        s.push_str(self.stimulus.duty_cycle.tag());
        push_q(&mut s, self.stimulus.phase_offset_ms, 10.0);
        push_q(&mut s, self.stimulus.duration_minutes, 10.0);
        s.push('|');
        // sensor_summary (RuView + optional EEG)
        push_q(&mut s, self.ruview.breathing_rate, 10.0);
        push_q(&mut s, self.ruview.breathing_stability, 100.0);
        push_q(&mut s, self.ruview.motion_artifact, 100.0);
        push_q(&mut s, self.ruview.stillness_score, 100.0);
        if let Some(e) = &self.eeg {
            push_q(&mut s, e.gamma_power_gain, 100.0);
            push_q(&mut s, e.phase_locking_value, 100.0);
            push_q(&mut s, e.artifact_score, 100.0);
        } else {
            s.push_str("no_eeg");
        }
        s.push('|');
        // response_summary
        push_q(&mut s, self.outcome.entrainment_score, 1000.0);
        push_q(&mut s, self.outcome.recommended_next_frequency_hz, 10.0);
        s.push(if self.outcome.safety_pass { 'P' } else { 'F' });
        s.push('|');
        // safety_events
        for ev in &self.safety_events {
            s.push_str(&serde_json::to_string(ev).unwrap_or_default());
            s.push(';');
        }
        s.into_bytes()
    }
}

/// Append a float quantized to `1/scale` resolution, as a stable integer token.
fn push_q(s: &mut String, v: f64, scale: f64) {
    let q = if v.is_finite() {
        (v * scale).round() as i64
    } else {
        i64::MIN
    };
    s.push_str(&q.to_string());
    s.push(',');
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safety::{AdverseEvent, StopReason};

    fn builder() -> SessionBuilder {
        SessionBuilder::new(
            "subject-A",
            VersionTriple::default(),
            1_700_000_000_000,
            StimulusParameters::prior(),
            RuViewState::calm_baseline(),
            SubjectiveReport::default(),
            Outcome {
                entrainment_score: 0.71,
                safety_pass: true,
                recommended_next_frequency_hz: 39.5,
            },
        )
    }

    #[test]
    fn identical_sessions_hash_identically() {
        let a = builder().finalize();
        let b = builder().finalize();
        assert_eq!(a.session_hash, b.session_hash);
        assert_eq!(a.session_id, b.session_id);
    }

    #[test]
    fn changing_frequency_changes_hash() {
        let a = builder().finalize();
        let mut b2 = builder();
        b2.stimulus.frequency_hz = 41.0;
        let b = b2.finalize();
        assert_ne!(a.session_hash, b.session_hash);
    }

    #[test]
    fn safety_events_alter_the_witness() {
        let a = builder().finalize();
        let b = builder()
            .with_safety_events(vec![StopReason::AdverseEvent(AdverseEvent::Dizziness)])
            .finalize();
        assert_ne!(a.session_hash, b.session_hash);
    }

    #[test]
    fn session_id_is_hash_prefix() {
        let r = builder().finalize();
        assert_eq!(r.session_id.0, r.session_hash[..16]);
    }

    #[test]
    fn record_roundtrips_through_json() {
        let r = builder().finalize();
        let json = serde_json::to_string(&r).unwrap();
        let back: SessionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn eeg_presence_changes_witness() {
        let a = builder().finalize();
        let b = builder()
            .with_eeg(EegMeasurement {
                gamma_power_gain: 0.4,
                phase_locking_value: 0.6,
                artifact_score: 0.03,
            })
            .finalize();
        assert_ne!(a.session_hash, b.session_hash);
    }
}
