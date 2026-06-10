//! Constrained optimizer — Phase 1 calibration, Phase 2 Bayesian optimization,
//! Phase 4 closed-loop control (ADR-250 §8).
//!
//! **Invariant (enforced by construction and asserted in tests):** every
//! [`StimulusParameters`] this module emits satisfies
//! [`SafetyEnvelope::contains`]. The optimizer searches *frequency* over the
//! envelope's 0.1 Hz grid (the ±0.1 Hz control spec, ADR-250 §18) while holding
//! intensity at conservative values — it never widens the envelope (ADR-250 §12).

use crate::math::{
    back_subst_transpose, cholesky, dot, forward_subst, normal_cdf, normal_pdf, rbf_kernel,
};
use crate::stimulus::{SafetyEnvelope, StimulusParameters};

/// Phase-1 conservative calibration sweep (ADR-250 §8): short sessions at each
/// integer Hz in the band. Hands its results to the Bayesian optimizer.
#[derive(Debug, Clone)]
pub struct CalibrationPlan {
    frequencies: Vec<f64>,
    next: usize,
}

impl CalibrationPlan {
    pub fn new(envelope: &SafetyEnvelope) -> Self {
        Self {
            frequencies: envelope.calibration_frequencies(),
            next: 0,
        }
    }

    /// Number of calibration sessions still pending.
    pub fn remaining(&self) -> usize {
        self.frequencies.len().saturating_sub(self.next)
    }

    /// The next calibration stimulus (short duration, conservative intensity),
    /// or `None` when the sweep is complete. Always inside the envelope.
    pub fn next_stimulus(
        &mut self,
        envelope: &SafetyEnvelope,
        session_minutes: f64,
    ) -> Option<StimulusParameters> {
        let f = *self.frequencies.get(self.next)?;
        self.next += 1;
        let mut s = StimulusParameters::prior();
        s.frequency_hz = f;
        s.duration_minutes = session_minutes;
        Some(envelope.clamp(s))
    }
}

/// Gaussian-process surrogate over the 1-D frequency axis with an
/// Expected-Improvement acquisition (ADR-250 §8 Phase 2).
#[derive(Debug, Clone)]
pub struct BayesianOptimizer {
    /// RBF length scale in Hz.
    pub length_scale: f64,
    /// GP signal variance.
    pub signal_var: f64,
    /// Observation noise variance (jitter; also keeps K SPD).
    pub noise_var: f64,
    /// EI exploration margin.
    pub xi: f64,
    /// Observed `(frequency_hz, score)` pairs.
    obs_x: Vec<f64>,
    obs_y: Vec<f64>,
}

impl Default for BayesianOptimizer {
    fn default() -> Self {
        Self {
            length_scale: 1.5,
            signal_var: 1.0,
            noise_var: 1e-4,
            xi: 0.01,
            obs_x: Vec::new(),
            obs_y: Vec::new(),
        }
    }
}

impl BayesianOptimizer {
    /// Record a calibration/optimization result.
    pub fn observe(&mut self, frequency_hz: f64, score: f64) {
        self.obs_x.push(frequency_hz);
        self.obs_y.push(score);
    }

    /// Number of observations so far.
    pub fn n_obs(&self) -> usize {
        self.obs_x.len()
    }

    /// Best observed score, or `None` if no observations.
    pub fn best(&self) -> Option<(f64, f64)> {
        let mut best: Option<(f64, f64)> = None;
        for (&x, &y) in self.obs_x.iter().zip(&self.obs_y) {
            if best.map(|(_, by)| y > by).unwrap_or(true) {
                best = Some((x, y));
            }
        }
        best
    }

    /// GP posterior `(mean, variance)` at `x`. Falls back to `(0, signal_var)`
    /// (the prior) when there are no observations or K is not SPD.
    pub fn predict(&self, x: f64) -> (f64, f64) {
        let n = self.obs_x.len();
        if n == 0 {
            return (0.0, self.signal_var);
        }
        // K = RBF(X,X) + noise·I
        let mut k = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut v = rbf_kernel(
                    &[self.obs_x[i]],
                    &[self.obs_x[j]],
                    self.length_scale,
                    self.signal_var,
                );
                if i == j {
                    v += self.noise_var;
                }
                k[i * n + j] = v;
            }
        }
        let l = match cholesky(&k, n) {
            Some(l) => l,
            None => return (0.0, self.signal_var),
        };
        // k* = RBF(X, x)
        let kstar: Vec<f64> = (0..n)
            .map(|i| rbf_kernel(&[self.obs_x[i]], &[x], self.length_scale, self.signal_var))
            .collect();
        // alpha = K^-1 y  (solve L Lᵀ alpha = y)
        let y1 = forward_subst(&l, &self.obs_y, n);
        let alpha = back_subst_transpose(&l, &y1, n);
        let mean = dot(&kstar, &alpha);
        // var = k(x,x) - v·v, where L v = k*
        let v = forward_subst(&l, &kstar, n);
        let var = (self.signal_var - dot(&v, &v)).max(0.0);
        (mean, var)
    }

    /// Expected Improvement (for maximization) at `x`.
    pub fn expected_improvement(&self, x: f64) -> f64 {
        let best = match self.best() {
            Some((_, by)) => by,
            None => return self.signal_var.sqrt(), // pure exploration
        };
        let (mu, var) = self.predict(x);
        let sigma = var.sqrt();
        if sigma <= 1e-12 {
            return 0.0;
        }
        let imp = mu - best - self.xi;
        let z = imp / sigma;
        imp * normal_cdf(z) + sigma * normal_pdf(z)
    }

    /// Recommend the next stimulus by maximizing EI over the envelope's 0.1 Hz
    /// grid, holding `base`'s intensity (clamped). The result is guaranteed
    /// inside the envelope. With no observations it returns the 40 Hz prior.
    pub fn recommend(
        &self,
        envelope: &SafetyEnvelope,
        base: &StimulusParameters,
    ) -> Recommendation {
        if self.obs_x.is_empty() {
            let s = envelope.clamp(*base);
            return Recommendation {
                stimulus: s,
                expected_improvement: 0.0,
                predicted_score: 0.0,
                confidence: 0.0,
            };
        }
        let grid = fine_grid(envelope);
        let mut best_f = base.frequency_hz;
        let mut best_ei = f64::NEG_INFINITY;
        for &f in &grid {
            let ei = self.expected_improvement(f);
            if ei > best_ei {
                best_ei = ei;
                best_f = f;
            }
        }
        let (mu, var) = self.predict(best_f);
        let mut s = *base;
        s.frequency_hz = best_f;
        let s = envelope.clamp(s);
        Recommendation {
            stimulus: s,
            expected_improvement: best_ei.max(0.0),
            predicted_score: mu,
            // Confidence shrinks with posterior variance (more data near the
            // pick → tighter → higher confidence), squashed to [0,1].
            confidence: 1.0 / (1.0 + var.sqrt()),
        }
    }
}

/// A recommendation plus its explainability fields (ADR-250 §14 response,
/// §16 "explainable recommendation").
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Recommendation {
    pub stimulus: StimulusParameters,
    pub expected_improvement: f64,
    pub predicted_score: f64,
    pub confidence: f64,
}

/// Phase-4 closed-loop controller (ADR-250 §8). Applies bounded mid-session
/// adjustments; every output is re-clamped to the envelope.
#[derive(Debug, Clone, Copy)]
pub struct ClosedLoopController {
    /// Max single-step frequency nudge in Hz (kept small for safety).
    pub max_freq_step_hz: f64,
    /// Entrainment below this triggers a corrective nudge.
    pub entrainment_floor: f64,
    /// Comfort below this triggers an intensity reduction.
    pub comfort_floor: f64,
    /// Multiplicative intensity reduction on discomfort.
    pub intensity_backoff: f64,
}

impl Default for ClosedLoopController {
    fn default() -> Self {
        Self {
            max_freq_step_hz: 0.5,
            entrainment_floor: 0.3,
            comfort_floor: 0.5,
            intensity_backoff: 0.8,
        }
    }
}

/// One closed-loop action recommendation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoopAction {
    /// Continue unchanged.
    Hold,
    /// Adjust to a new (already envelope-clamped) stimulus.
    Adjust(StimulusParameters),
}

impl ClosedLoopController {
    /// Decide the next in-session action from live entrainment/comfort.
    /// `gradient_sign` indicates which frequency direction recently improved
    /// entrainment (+1 up, −1 down, 0 unknown); the nudge respects it.
    pub fn step(
        &self,
        envelope: &SafetyEnvelope,
        current: &StimulusParameters,
        live_entrainment: f64,
        live_comfort: f64,
        gradient_sign: f64,
    ) -> LoopAction {
        // Comfort first: discomfort always reduces intensity (safety-leaning).
        if live_comfort < self.comfort_floor {
            let mut s = *current;
            s.brightness_level *= self.intensity_backoff;
            s.volume_level *= self.intensity_backoff;
            return LoopAction::Adjust(envelope.clamp(s));
        }
        // Entrainment fading: small bounded frequency nudge toward improvement.
        if live_entrainment < self.entrainment_floor {
            let dir = if gradient_sign >= 0.0 { 1.0 } else { -1.0 };
            let mut s = *current;
            s.frequency_hz += dir * self.max_freq_step_hz;
            return LoopAction::Adjust(envelope.clamp(s));
        }
        LoopAction::Hold
    }
}

/// The 0.1 Hz candidate grid over the envelope (ADR-250 §18 ±0.1 Hz precision).
fn fine_grid(envelope: &SafetyEnvelope) -> Vec<f64> {
    let lo = (envelope.min_hz * 10.0).round() as i64;
    let hi = (envelope.max_hz * 10.0).round() as i64;
    (lo..=hi).map(|i| i as f64 / 10.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_sweep_covers_band_and_stays_safe() {
        let env = SafetyEnvelope::conservative();
        let mut plan = CalibrationPlan::new(&env);
        let mut seen = Vec::new();
        while let Some(s) = plan.next_stimulus(&env, 5.0) {
            assert!(env.contains(&s));
            seen.push(s.frequency_hz);
        }
        assert_eq!(seen.first(), Some(&36.0));
        assert_eq!(seen.last(), Some(&44.0));
        assert_eq!(plan.remaining(), 0);
    }

    #[test]
    fn gp_recovers_a_quadratic_peak() {
        // Synthetic score surface peaked at 39.5 Hz.
        let env = SafetyEnvelope::conservative();
        let mut bo = BayesianOptimizer::default();
        let truth = |f: f64| 1.0 - 0.05 * (f - 39.5).powi(2);
        for f in env.calibration_frequencies() {
            bo.observe(f, truth(f));
        }
        let rec = bo.recommend(&env, &StimulusParameters::prior());
        assert!(env.contains(&rec.stimulus));
        // Should land within ±1 Hz of the true peak (ADR-250 §18 repeatability).
        assert!((rec.stimulus.frequency_hz - 39.5).abs() <= 1.0);
    }

    #[test]
    fn recommendation_is_always_in_envelope() {
        let env = SafetyEnvelope::conservative();
        let mut bo = BayesianOptimizer::default();
        // Adversarial: pretend the band edge is best.
        for f in env.calibration_frequencies() {
            bo.observe(f, if f >= 44.0 { 10.0 } else { 0.0 });
        }
        let rec = bo.recommend(&env, &StimulusParameters::prior());
        assert!(env.contains(&rec.stimulus));
        assert!(rec.stimulus.frequency_hz <= env.max_hz);
    }

    #[test]
    fn no_observations_returns_prior() {
        let env = SafetyEnvelope::conservative();
        let bo = BayesianOptimizer::default();
        let rec = bo.recommend(&env, &StimulusParameters::prior());
        assert_eq!(rec.stimulus.frequency_hz, 40.0);
    }

    #[test]
    fn closed_loop_reduces_intensity_on_discomfort() {
        let env = SafetyEnvelope::conservative();
        let ctl = ClosedLoopController::default();
        let cur = StimulusParameters::prior();
        match ctl.step(&env, &cur, 0.6, 0.2, 0.0) {
            LoopAction::Adjust(s) => {
                assert!(s.brightness_level < cur.brightness_level);
                assert!(env.contains(&s));
            }
            LoopAction::Hold => panic!("should have adjusted intensity"),
        }
    }

    #[test]
    fn closed_loop_nudge_stays_in_envelope() {
        let env = SafetyEnvelope::conservative();
        let ctl = ClosedLoopController::default();
        let mut cur = StimulusParameters::prior();
        cur.frequency_hz = 43.8; // near the upper edge
        match ctl.step(&env, &cur, 0.1, 0.9, 1.0) {
            LoopAction::Adjust(s) => assert!(env.contains(&s)),
            LoopAction::Hold => {}
        }
    }

    #[test]
    fn closed_loop_holds_when_healthy() {
        let env = SafetyEnvelope::conservative();
        let ctl = ClosedLoopController::default();
        let cur = StimulusParameters::prior();
        assert_eq!(ctl.step(&env, &cur, 0.8, 0.9, 0.0), LoopAction::Hold);
    }
}
