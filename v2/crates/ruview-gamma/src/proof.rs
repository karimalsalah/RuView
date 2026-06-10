//! Deterministic proof bundle (mirrors the `nvsim` / `verify.py` pattern).
//!
//! Runs a fixed reference participant through the full governed pipeline
//! (enroll → calibration sweep → witnessed audit) and hashes the resulting
//! session witnesses into a single bundle digest. If the digest matches
//! [`Proof::EXPECTED_WITNESS`], the optimizer math, simulator physics, response
//! update, and session-hashing code paths are all byte-identical to the
//! published reference. Any silent drift in any of them shifts the digest and
//! the test fails loudly.

use crate::ruflo::{Consent, RufloGovernor};
use crate::response::RuViewState;
use crate::simulator::{stable_hash, LatentPerson, ResponseSimulator};
use crate::stimulus::SafetyEnvelope;

/// Deterministic-proof harness for `ruview-gamma`.
pub struct Proof;

impl Proof {
    /// Reference participant id (drives the latent physiology).
    pub const PERSON_ID: &'static str = "reference-subject-000";

    /// Reference simulator seed.
    pub const SEED: u64 = 42;

    /// SHA-256 (hex) over the concatenated session witnesses of the reference
    /// calibration run. Pinned so CI catches any drift.
    pub const EXPECTED_WITNESS: &'static str =
        "13cb164cc3b3b02da8cdfbb5c23fdd07431c58498396d75a3d9a470305981758";

    /// Run the reference scenario and return its bundle witness (hex SHA-256).
    pub fn reference_witness() -> String {
        let envelope = SafetyEnvelope::conservative();
        let mut gov = RufloGovernor::enroll(Self::PERSON_ID, envelope, &[], Consent::Granted)
            .expect("reference participant enrolls cleanly");
        let sim = ResponseSimulator::new(Self::SEED);
        let latent = LatentPerson::from_id(Self::PERSON_ID);
        let state = RuViewState::calm_baseline();
        gov.run_calibration(&sim, &latent, &state, 5.0, 1_700_000_000_000)
            .expect("reference calibration runs");

        // Concatenate every session witness in order, then hash the bundle.
        let mut chunks: Vec<&[u8]> = Vec::new();
        for rec in gov.audit_log() {
            chunks.push(rec.session_hash.as_bytes());
        }
        let digest = stable_hash(&chunks);
        hex(&digest)
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_witness_is_deterministic() {
        assert_eq!(Proof::reference_witness(), Proof::reference_witness());
    }

    #[test]
    fn reference_witness_matches_expected() {
        // If this fails after an intentional change, regenerate the constant
        // from the test output and document the change in CHANGELOG.
        assert_eq!(Proof::reference_witness(), Proof::EXPECTED_WITNESS);
    }
}
