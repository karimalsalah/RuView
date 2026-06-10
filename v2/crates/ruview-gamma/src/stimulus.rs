//! Stimulus parameters and the safety envelope (ADR-250 §5, §12).
//!
//! The [`SafetyEnvelope`] is the load-bearing safety primitive of the whole
//! crate: **no recommendation, calibration step, or closed-loop nudge may ever
//! produce a [`StimulusParameters`] that fails [`SafetyEnvelope::validate`].**
//! Every code path that emits a stimulus setting routes through
//! [`SafetyEnvelope::clamp`] (best-effort coercion) and is asserted against
//! [`SafetyEnvelope::contains`] in tests.

use serde::{Deserialize, Serialize};

use crate::math::clamp_safe;

/// Stimulation modality (ADR-250 §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Sound only.
    Audio,
    /// Light only.
    Visual,
    /// Combined audio-visual — GENUS-style, the preferred protocol.
    AudioVisual,
}

impl Modality {
    /// Canonical lowercase tag used in the session witness.
    pub fn tag(self) -> &'static str {
        match self {
            Modality::Audio => "audio",
            Modality::Visual => "visual",
            Modality::AudioVisual => "audio_visual",
        }
    }
}

/// Duty-cycle shape (ADR-250 §5). Conservative ordering: `Continuous` first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DutyCycle {
    /// Steady stimulation for the whole session.
    Continuous,
    /// Amplitude ramps up/down — gentlest onset.
    Ramped,
    /// On/off pulsing — explored only after tolerance is established.
    Pulsed,
}

impl DutyCycle {
    pub fn tag(self) -> &'static str {
        match self {
            DutyCycle::Continuous => "continuous",
            DutyCycle::Ramped => "ramped",
            DutyCycle::Pulsed => "pulsed",
        }
    }
}

/// A concrete stimulation setting. All intensity fields are normalized `[0,1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StimulusParameters {
    /// Carrier/flicker frequency in Hz (search band 36–44, prior 40).
    pub frequency_hz: f64,
    /// Stimulation modality.
    pub modality: Modality,
    /// Visual brightness in `[0,1]` (capped well below unsafe flicker intensity).
    pub brightness_level: f64,
    /// Audio volume in `[0,1]` (comfort-bounded).
    pub volume_level: f64,
    /// Duty-cycle shape.
    pub duty_cycle: DutyCycle,
    /// Inter-modality phase offset in milliseconds.
    pub phase_offset_ms: f64,
    /// Session duration in minutes.
    pub duration_minutes: f64,
}

impl StimulusParameters {
    /// The evidence-based starting prior: 40 Hz combined audio-visual, gentle
    /// intensities, continuous, short (ADR-250 §5 "Starting prior").
    pub fn prior() -> Self {
        Self {
            frequency_hz: 40.0,
            modality: Modality::AudioVisual,
            brightness_level: 0.30,
            volume_level: 0.28,
            duty_cycle: DutyCycle::Continuous,
            phase_offset_ms: 0.0,
            duration_minutes: 10.0,
        }
    }
}

/// Reasons a [`StimulusParameters`] is rejected by the envelope. Each variant
/// is logged verbatim into the RuFlo safety trail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "detail")]
pub enum EnvelopeViolation {
    /// Frequency outside `[min_hz, max_hz]`.
    Frequency { value: f64, min: f64, max: f64 },
    /// Brightness above the hard cap.
    Brightness { value: f64, cap: f64 },
    /// Volume above the hard cap.
    Volume { value: f64, cap: f64 },
    /// Duration above the per-stage maximum.
    Duration { value: f64, max: f64 },
    /// A non-finite (NaN/Inf) field was supplied.
    NonFinite { field: &'static str },
}

/// The predefined safety envelope. Optimization happens **only inside** these
/// bounds; the system "must never autonomously expand beyond the allowed safety
/// envelope" (ADR-250 §12). The envelope itself is data, never widened by the
/// optimizer — only an operator constructs a wider one deliberately.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub min_hz: f64,
    pub max_hz: f64,
    /// Hard brightness cap — flicker-exposure conservatism (ADR-250 §12, OQ 7).
    pub brightness_cap: f64,
    /// Hard volume cap — comfort conservatism.
    pub volume_cap: f64,
    /// Maximum session duration for the current stage.
    pub max_duration_minutes: f64,
    /// Maximum absolute inter-modality phase offset.
    pub max_phase_offset_ms: f64,
}

impl SafetyEnvelope {
    /// Conservative research-grade default envelope (ADR-250 §5 default ranges,
    /// "safety_profile: conservative").
    pub fn conservative() -> Self {
        Self {
            min_hz: 36.0,
            max_hz: 44.0,
            brightness_cap: 0.40,
            volume_cap: 0.40,
            max_duration_minutes: 15.0,
            max_phase_offset_ms: 5.0,
        }
    }

    /// `true` iff every field of `s` lies inside the envelope and is finite.
    pub fn contains(&self, s: &StimulusParameters) -> bool {
        self.validate(s).is_ok()
    }

    /// Validate a setting, returning every violation found (not just the first)
    /// so the safety log is complete.
    pub fn validate(&self, s: &StimulusParameters) -> Result<(), Vec<EnvelopeViolation>> {
        let mut v = Vec::new();
        for (field, val) in [
            ("frequency_hz", s.frequency_hz),
            ("brightness_level", s.brightness_level),
            ("volume_level", s.volume_level),
            ("duration_minutes", s.duration_minutes),
            ("phase_offset_ms", s.phase_offset_ms),
        ] {
            if !val.is_finite() {
                v.push(EnvelopeViolation::NonFinite { field });
            }
        }
        if !v.is_empty() {
            return Err(v);
        }
        if s.frequency_hz < self.min_hz || s.frequency_hz > self.max_hz {
            v.push(EnvelopeViolation::Frequency {
                value: s.frequency_hz,
                min: self.min_hz,
                max: self.max_hz,
            });
        }
        if s.brightness_level > self.brightness_cap || s.brightness_level < 0.0 {
            v.push(EnvelopeViolation::Brightness {
                value: s.brightness_level,
                cap: self.brightness_cap,
            });
        }
        if s.volume_level > self.volume_cap || s.volume_level < 0.0 {
            v.push(EnvelopeViolation::Volume {
                value: s.volume_level,
                cap: self.volume_cap,
            });
        }
        if s.duration_minutes > self.max_duration_minutes || s.duration_minutes <= 0.0 {
            v.push(EnvelopeViolation::Duration {
                value: s.duration_minutes,
                max: self.max_duration_minutes,
            });
        }
        if v.is_empty() {
            Ok(())
        } else {
            Err(v)
        }
    }

    /// Best-effort coercion of `s` into the envelope. Used as a defensive final
    /// stage on any emitted recommendation; coercion can only ever *reduce*
    /// intensity / pull frequency inward, never expand it. The result always
    /// satisfies [`contains`](Self::contains).
    pub fn clamp(&self, mut s: StimulusParameters) -> StimulusParameters {
        s.frequency_hz = clamp_safe(s.frequency_hz, self.min_hz, self.max_hz);
        s.brightness_level = clamp_safe(s.brightness_level, 0.0, self.brightness_cap);
        s.volume_level = clamp_safe(s.volume_level, 0.0, self.volume_cap);
        s.phase_offset_ms =
            clamp_safe(s.phase_offset_ms, -self.max_phase_offset_ms, self.max_phase_offset_ms);
        // Duration must be strictly positive; floor at 1 minute.
        s.duration_minutes = clamp_safe(s.duration_minutes, 1.0, self.max_duration_minutes);
        s
    }

    /// The calibration sweep grid (ADR-250 §8 Phase 1): integer-Hz steps across
    /// the band, intersected with the envelope. Returns frequencies in Hz.
    pub fn calibration_frequencies(&self) -> Vec<f64> {
        let lo = self.min_hz.ceil() as i32;
        let hi = self.max_hz.floor() as i32;
        (lo..=hi).map(|f| f as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prior_is_inside_conservative_envelope() {
        let env = SafetyEnvelope::conservative();
        assert!(env.contains(&StimulusParameters::prior()));
    }

    #[test]
    fn frequency_outside_band_is_rejected() {
        let env = SafetyEnvelope::conservative();
        let mut s = StimulusParameters::prior();
        s.frequency_hz = 50.0;
        let err = env.validate(&s).unwrap_err();
        assert!(matches!(err[0], EnvelopeViolation::Frequency { .. }));
    }

    #[test]
    fn brightness_above_cap_is_rejected() {
        let env = SafetyEnvelope::conservative();
        let mut s = StimulusParameters::prior();
        s.brightness_level = 0.9;
        assert!(!env.contains(&s));
    }

    #[test]
    fn clamp_output_is_always_contained() {
        let env = SafetyEnvelope::conservative();
        let hostile = StimulusParameters {
            frequency_hz: 1000.0,
            modality: Modality::Visual,
            brightness_level: 5.0,
            volume_level: -2.0,
            duty_cycle: DutyCycle::Pulsed,
            phase_offset_ms: 999.0,
            duration_minutes: 1e6,
        };
        assert!(env.contains(&env.clamp(hostile)));
    }

    #[test]
    fn clamp_neutralizes_nan() {
        let env = SafetyEnvelope::conservative();
        let mut s = StimulusParameters::prior();
        s.frequency_hz = f64::NAN;
        s.brightness_level = f64::INFINITY;
        let c = env.clamp(s);
        assert!(env.contains(&c));
        assert_eq!(c.frequency_hz, env.min_hz);
        assert_eq!(c.brightness_level, 0.0);
    }

    #[test]
    fn calibration_grid_is_36_to_44() {
        let env = SafetyEnvelope::conservative();
        let grid = env.calibration_frequencies();
        assert_eq!(grid.first(), Some(&36.0));
        assert_eq!(grid.last(), Some(&44.0));
        assert_eq!(grid.len(), 9);
    }
}
