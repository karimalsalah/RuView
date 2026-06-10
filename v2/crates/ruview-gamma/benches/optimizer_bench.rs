//! Criterion benchmarks for the adaptive-gamma governed loop (ADR-250 §17).
//!
//! Measures the latency-sensitive paths: a full calibration sweep, a single
//! Bayesian recommendation, a closed-loop safety tick, and a bandit decision.
//! The safety-stop tick is the figure compared against ADR-250 §17's < 500 ms
//! bound — it is O(1) and lands far below.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ruview_gamma::bandit::{BanditContext, ContextualBandit};
use ruview_gamma::optimizer::BayesianOptimizer;
use ruview_gamma::response::RuViewState;
use ruview_gamma::ruflo::{Consent, RufloGovernor};
use ruview_gamma::safety::{SafetyMonitor, SafetyTick};
use ruview_gamma::simulator::{LatentPerson, ResponseSimulator};
use ruview_gamma::stimulus::{SafetyEnvelope, StimulusParameters};

fn bench_calibration(c: &mut Criterion) {
    let env = SafetyEnvelope::conservative();
    let sim = ResponseSimulator::new(42);
    let latent = LatentPerson::from_id("bench-subject");
    let state = RuViewState::calm_baseline();
    c.bench_function("gamma_calibration_sweep", |b| {
        b.iter(|| {
            let mut gov =
                RufloGovernor::enroll("bench-subject", env, &[], Consent::Granted).unwrap();
            gov.run_calibration(black_box(&sim), &latent, &state, 5.0, 0)
                .unwrap();
            black_box(gov.audit_log().len())
        })
    });
}

fn bench_recommend(c: &mut Criterion) {
    let env = SafetyEnvelope::conservative();
    let mut bo = BayesianOptimizer::default();
    for f in env.calibration_frequencies() {
        bo.observe(f, 1.0 - 0.05 * (f - 39.5).powi(2));
    }
    let base = StimulusParameters::prior();
    c.bench_function("gamma_bayesian_recommend", |b| {
        b.iter(|| black_box(bo.recommend(black_box(&env), black_box(&base))))
    });
}

fn bench_safety_tick(c: &mut Criterion) {
    c.bench_function("gamma_safety_tick", |b| {
        b.iter(|| {
            let mut m = SafetyMonitor::default();
            black_box(m.evaluate(black_box(SafetyTick {
                adverse: None,
                sensor_confidence: 0.9,
                stimulus_in_envelope: true,
            })))
        })
    });
}

fn bench_bandit(c: &mut Criterion) {
    let env = SafetyEnvelope::conservative();
    let candidates: Vec<StimulusParameters> = [38.0, 40.0, 42.0]
        .iter()
        .map(|&f| {
            let mut s = StimulusParameters::prior();
            s.frequency_hz = f;
            s
        })
        .collect();
    let bandit = ContextualBandit::new(&env, &candidates, 1.0).unwrap();
    let ctx = BanditContext {
        sleep_quality: 0.7,
        time_of_day: 0.5,
        breathing_state: 0.8,
        motion_state: 0.1,
        fatigue_proxy: 0.2,
        prior_response: 0.6,
    };
    c.bench_function("gamma_bandit_select", |b| {
        b.iter(|| black_box(bandit.select(black_box(&ctx))))
    });
}

criterion_group!(
    benches,
    bench_calibration,
    bench_recommend,
    bench_safety_tick,
    bench_bandit
);
criterion_main!(benches);
