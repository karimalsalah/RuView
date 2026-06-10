//! Phase-3 contextual bandit (ADR-250 §8).
//!
//! Once enough sessions exist, protocol selection becomes state-dependent:
//! `context = [sleep_quality, time_of_day, breathing_state, motion_state,
//! fatigue_proxy, prior_response]` → `action = stimulus setting` →
//! `reward = safe_entrainment_score`. We use **LinUCB** (disjoint linear model
//! per arm) — small, deterministic, explainable, and edge-deployable.
//!
//! Arms are a discrete set of *envelope-safe* stimulus settings supplied by the
//! caller; the bandit never invents an out-of-envelope action because it can
//! only ever return one of the arms it was given.

use crate::math::clamp_safe;
use crate::stimulus::{SafetyEnvelope, StimulusParameters};

/// Context vector dimensionality (ADR-250 §8 Phase 3 context list).
pub const CONTEXT_DIM: usize = 6;

/// The decision context, normalized to `[0,1]` per field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BanditContext {
    pub sleep_quality: f64,
    pub time_of_day: f64,
    pub breathing_state: f64,
    pub motion_state: f64,
    pub fatigue_proxy: f64,
    pub prior_response: f64,
}

impl BanditContext {
    /// Flat feature vector in the documented field order.
    pub fn features(&self) -> [f64; CONTEXT_DIM] {
        [
            clamp_safe(self.sleep_quality, 0.0, 1.0),
            clamp_safe(self.time_of_day, 0.0, 1.0),
            clamp_safe(self.breathing_state, 0.0, 1.0),
            clamp_safe(self.motion_state, 0.0, 1.0),
            clamp_safe(self.fatigue_proxy, 0.0, 1.0),
            clamp_safe(self.prior_response, 0.0, 1.0),
        ]
    }
}

/// One LinUCB arm: a fixed safe stimulus plus its online linear model. The
/// per-arm `A⁻¹` is maintained incrementally via the Sherman–Morrison update,
/// so no matrix inversion runs at decision time.
#[derive(Debug, Clone)]
struct Arm {
    stimulus: StimulusParameters,
    /// A⁻¹ (d×d, row-major), initialized to I.
    a_inv: [f64; CONTEXT_DIM * CONTEXT_DIM],
    /// b (d), initialized to 0.
    b: [f64; CONTEXT_DIM],
}

impl Arm {
    fn new(stimulus: StimulusParameters) -> Self {
        let mut a_inv = [0.0; CONTEXT_DIM * CONTEXT_DIM];
        for i in 0..CONTEXT_DIM {
            a_inv[i * CONTEXT_DIM + i] = 1.0;
        }
        Self {
            stimulus,
            a_inv,
            b: [0.0; CONTEXT_DIM],
        }
    }

    /// theta = A⁻¹ b
    fn theta(&self) -> [f64; CONTEXT_DIM] {
        mat_vec(&self.a_inv, &self.b)
    }

    /// UCB score: μ + α·√(xᵀ A⁻¹ x).
    fn ucb(&self, x: &[f64; CONTEXT_DIM], alpha: f64) -> f64 {
        let theta = self.theta();
        let mean: f64 = theta.iter().zip(x).map(|(t, xi)| t * xi).sum();
        let ainv_x = mat_vec(&self.a_inv, x);
        let var: f64 = x.iter().zip(&ainv_x).map(|(xi, v)| xi * v).sum();
        mean + alpha * var.max(0.0).sqrt()
    }

    /// Online update with observed `(x, reward)` via Sherman–Morrison:
    /// A ← A + x xᵀ  ⇒  A⁻¹ ← A⁻¹ − (A⁻¹ x xᵀ A⁻¹)/(1 + xᵀ A⁻¹ x).
    fn update(&mut self, x: &[f64; CONTEXT_DIM], reward: f64) {
        let ainv_x = mat_vec(&self.a_inv, x); // A⁻¹ x  (d)
        let denom = 1.0 + x.iter().zip(&ainv_x).map(|(xi, v)| xi * v).sum::<f64>();
        // A⁻¹ ← A⁻¹ − (ainv_x)(ainv_x)ᵀ / denom   (since A symmetric ⇒ xᵀA⁻¹ = (A⁻¹x)ᵀ)
        for i in 0..CONTEXT_DIM {
            for j in 0..CONTEXT_DIM {
                self.a_inv[i * CONTEXT_DIM + j] -= ainv_x[i] * ainv_x[j] / denom;
            }
        }
        for i in 0..CONTEXT_DIM {
            self.b[i] += reward * x[i];
        }
    }
}

/// LinUCB contextual bandit over a fixed set of envelope-safe arms.
#[derive(Debug, Clone)]
pub struct ContextualBandit {
    arms: Vec<Arm>,
    /// Exploration coefficient α.
    pub alpha: f64,
}

impl ContextualBandit {
    /// Build a bandit from candidate stimuli. Each candidate is **clamped into
    /// the envelope** on the way in, so no arm can ever be unsafe (ADR-250 §12).
    /// Returns `None` if no candidates were supplied.
    pub fn new(envelope: &SafetyEnvelope, candidates: &[StimulusParameters], alpha: f64) -> Option<Self> {
        if candidates.is_empty() {
            return None;
        }
        let arms = candidates
            .iter()
            .map(|s| Arm::new(envelope.clamp(*s)))
            .collect();
        Some(Self { arms, alpha })
    }

    /// Number of arms.
    pub fn n_arms(&self) -> usize {
        self.arms.len()
    }

    /// Select the arm index with the highest UCB for `ctx`. Deterministic
    /// tie-break: lowest index wins.
    pub fn select(&self, ctx: &BanditContext) -> usize {
        let x = ctx.features();
        let mut best_i = 0;
        let mut best = f64::NEG_INFINITY;
        for (i, arm) in self.arms.iter().enumerate() {
            let u = arm.ucb(&x, self.alpha);
            if u > best {
                best = u;
                best_i = i;
            }
        }
        best_i
    }

    /// The stimulus for an arm index (always envelope-safe).
    pub fn stimulus(&self, arm: usize) -> StimulusParameters {
        self.arms[arm].stimulus
    }

    /// Record the reward for a chosen arm under context `ctx`.
    pub fn update(&mut self, arm: usize, ctx: &BanditContext, reward: f64) {
        let x = ctx.features();
        self.arms[arm].update(&x, reward);
    }
}

/// y = M x for row-major `d×d` M and length-`d` x.
fn mat_vec(m: &[f64; CONTEXT_DIM * CONTEXT_DIM], x: &[f64; CONTEXT_DIM]) -> [f64; CONTEXT_DIM] {
    let mut y = [0.0; CONTEXT_DIM];
    for i in 0..CONTEXT_DIM {
        let mut s = 0.0;
        for j in 0..CONTEXT_DIM {
            s += m[i * CONTEXT_DIM + j] * x[j];
        }
        y[i] = s;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stimulus::StimulusParameters;

    fn candidates() -> Vec<StimulusParameters> {
        [38.0, 40.0, 42.0]
            .iter()
            .map(|&f| {
                let mut s = StimulusParameters::prior();
                s.frequency_hz = f;
                s
            })
            .collect()
    }

    fn ctx(sleep: f64) -> BanditContext {
        BanditContext {
            sleep_quality: sleep,
            time_of_day: 0.5,
            breathing_state: 0.8,
            motion_state: 0.1,
            fatigue_proxy: 0.2,
            prior_response: 0.6,
        }
    }

    #[test]
    fn rejects_empty_candidates() {
        let env = SafetyEnvelope::conservative();
        assert!(ContextualBandit::new(&env, &[], 1.0).is_none());
    }

    #[test]
    fn all_arms_are_envelope_safe_even_if_candidate_is_not() {
        let env = SafetyEnvelope::conservative();
        let mut bad = StimulusParameters::prior();
        bad.frequency_hz = 100.0;
        bad.brightness_level = 9.0;
        let b = ContextualBandit::new(&env, &[bad], 1.0).unwrap();
        assert!(env.contains(&b.stimulus(0)));
    }

    #[test]
    fn learns_to_prefer_rewarded_arm() {
        let env = SafetyEnvelope::conservative();
        let mut b = ContextualBandit::new(&env, &candidates(), 0.1).unwrap();
        let c = ctx(0.9);
        // Arm 2 (42 Hz) is consistently best in this context.
        for _ in 0..50 {
            for arm in 0..b.n_arms() {
                let reward = if arm == 2 { 1.0 } else { 0.1 };
                b.update(arm, &c, reward);
            }
        }
        assert_eq!(b.select(&c), 2);
    }

    #[test]
    fn selection_is_deterministic() {
        let env = SafetyEnvelope::conservative();
        let b = ContextualBandit::new(&env, &candidates(), 1.0).unwrap();
        let c = ctx(0.5);
        assert_eq!(b.select(&c), b.select(&c));
    }
}
