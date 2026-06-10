//! Small, dependency-light deterministic numerics shared across modules.
//!
//! All functions are pure `f64` and deterministic — no global state, no
//! randomness. Kept intentionally tiny so the Gaussian-process surrogate
//! (`optimizer.rs`) and the LinUCB bandit (`bandit.rs`) do not pull in a
//! linear-algebra crate that could perturb the witness with its own float
//! reassociation choices.

/// Standard-normal PDF φ(z).
#[inline]
pub fn normal_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

/// Error function via Abramowitz & Stegun 7.1.26 (max abs error ≈ 1.5e-7).
///
/// Deterministic and branch-stable, so the witness is portable across targets.
#[inline]
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Standard-normal CDF Φ(z).
#[inline]
pub fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

/// Dot product. Panics if lengths differ (caller invariant).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Squared-exponential (RBF) kernel k(a,b) = σ_f² · exp(−‖a−b‖²/(2ℓ²)).
#[inline]
pub fn rbf_kernel(a: &[f64], b: &[f64], length_scale: f64, signal_var: f64) -> f64 {
    let d2: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    signal_var * (-0.5 * d2 / (length_scale * length_scale)).exp()
}

/// Cholesky factor L (lower-triangular) of a symmetric positive-definite
/// matrix `a` stored row-major `n×n`. Returns `None` if not SPD (a non-positive
/// pivot), which the caller treats as "fall back to the prior mean".
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `L y = b` (forward substitution) for lower-triangular `L` (`n×n`).
pub fn forward_subst(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    y
}

/// Solve `Lᵀ x = y` (back substitution) for lower-triangular `L` (`n×n`).
pub fn back_subst_transpose(l: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Clamp helper that is explicit about non-finite handling: any non-finite
/// input (NaN **or** ±∞) maps to `lo`, so a degenerate value can never escape
/// the safety envelope upward toward `hi`.
#[inline]
pub fn clamp_safe(v: f64, lo: f64, hi: f64) -> f64 {
    if !v.is_finite() {
        lo
    } else {
        v.max(lo).min(hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn erf_known_values() {
        assert_abs_diff_eq!(erf(0.0), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(erf(1.0), 0.842_700_79, epsilon = 1e-6);
        assert_abs_diff_eq!(erf(-1.0), -0.842_700_79, epsilon = 1e-6);
    }

    #[test]
    fn normal_cdf_symmetry() {
        assert_abs_diff_eq!(normal_cdf(0.0), 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(normal_cdf(1.96), 0.975, epsilon = 1e-3);
        assert_abs_diff_eq!(normal_cdf(-1.96), 0.025, epsilon = 1e-3);
    }

    #[test]
    fn cholesky_solves_spd_system() {
        // A = [[4,2],[2,3]], b=[1,1] -> x = A^-1 b
        let a = vec![4.0, 2.0, 2.0, 3.0];
        let l = cholesky(&a, 2).expect("SPD");
        let b = vec![1.0, 1.0];
        let y = forward_subst(&l, &b, 2);
        let x = back_subst_transpose(&l, &y, 2);
        // Verify A x = b
        assert_abs_diff_eq!(4.0 * x[0] + 2.0 * x[1], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(2.0 * x[0] + 3.0 * x[1], 1.0, epsilon = 1e-9);
    }

    #[test]
    fn cholesky_rejects_non_spd() {
        // Indefinite matrix
        let a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    #[test]
    fn clamp_safe_maps_non_finite_to_low() {
        assert_eq!(clamp_safe(f64::NAN, 0.1, 0.9), 0.1);
        assert_eq!(clamp_safe(f64::INFINITY, 0.1, 0.9), 0.1);
        assert_eq!(clamp_safe(f64::NEG_INFINITY, 0.1, 0.9), 0.1);
        assert_eq!(clamp_safe(1.5, 0.1, 0.9), 0.9);
        assert_eq!(clamp_safe(-1.0, 0.1, 0.9), 0.1);
    }
}
