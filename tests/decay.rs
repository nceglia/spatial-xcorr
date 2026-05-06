use approx::assert_relative_eq;

use spatial_xcorr::fit_exponential_decay;

const BANDWIDTHS: [f32; 5] = [10.0, 20.0, 40.0, 80.0, 160.0];

fn synth_exp(amplitude: f32, lambda: f32) -> [f32; 5] {
    let mut out = [0.0f32; 5];
    for (i, &h) in BANDWIDTHS.iter().enumerate() {
        out[i] = amplitude * (-h / lambda).exp();
    }
    out
}

#[test]
fn fit_clean_exponential() {
    let scores = synth_exp(0.8, 50.0);
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert_relative_eq!(fit.decay_length, 50.0, epsilon = 1e-2);
    assert_relative_eq!(fit.amplitude, 0.8, epsilon = 1e-2);
    assert!(
        fit.r_squared > 0.999,
        "expected r² > 0.999, got {}",
        fit.r_squared
    );
    assert_eq!(fit.sign, 1);
}

#[test]
fn fit_negative_amplitude() {
    let positive = synth_exp(0.5, 30.0);
    let scores: Vec<f32> = positive.iter().map(|s| -s).collect();
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert_relative_eq!(fit.decay_length, 30.0, epsilon = 1e-2);
    assert_relative_eq!(fit.amplitude, 0.5, epsilon = 1e-2);
    assert_eq!(fit.sign, -1);
    assert!(
        fit.r_squared > 0.999,
        "expected r² > 0.999, got {}",
        fit.r_squared
    );
}

#[test]
fn fit_flat_curve_returns_nan_lambda() {
    let scores = [0.3_f32, 0.3, 0.3, 0.3, 0.3];
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert!(fit.decay_length.is_nan(), "λ={}", fit.decay_length);
    assert_eq!(fit.sign, 1);
    assert_relative_eq!(fit.amplitude, 0.3, epsilon = 1e-5);
}

#[test]
fn fit_growing_curve_returns_nan_lambda() {
    let scores = [0.1_f32, 0.2, 0.3, 0.4, 0.5];
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert!(fit.decay_length.is_nan(), "λ={}", fit.decay_length);
    assert!(
        fit.r_squared.is_finite(),
        "r² should be finite for a non-degenerate fit, got {}",
        fit.r_squared
    );
}

#[test]
fn fit_too_few_points() {
    let scores = [0.5_f32, f32::NAN, f32::NAN, f32::NAN, f32::NAN];
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert!(fit.decay_length.is_nan());
    assert!(fit.r_squared.is_nan());
    assert_eq!(fit.sign, 0);
}

/// Three finite points: (10, 0.5), (40, 0.3), (160, 0.1). Hand-computing the
/// log-linear fit:
///   y = ln|score| = [-0.6931, -1.2040, -2.3026]
///   x̄ = 70, ȳ ≈ -1.3999
///   slope ≈ -129.528 / 12600 ≈ -0.010280  →  λ = -1/slope ≈ 97.3
#[test]
fn fit_with_nan_inputs() {
    let scores = [0.5_f32, f32::NAN, 0.3, f32::NAN, 0.1];
    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);

    assert!(
        fit.decay_length.is_finite(),
        "expected finite λ, got {}",
        fit.decay_length
    );

    // Reproduce the analytical slope from the three surviving points.
    let x = [10.0_f32, 40.0, 160.0];
    let y = [0.5_f32.ln(), 0.3_f32.ln(), 0.1_f32.ln()];
    let mean_x = x.iter().sum::<f32>() / 3.0;
    let mean_y = y.iter().sum::<f32>() / 3.0;
    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    for (xi, yi) in x.iter().zip(y.iter()) {
        cov += (xi - mean_x) * (yi - mean_y);
        var_x += (xi - mean_x).powi(2);
    }
    let expected_slope = cov / var_x;
    let expected_lambda = -1.0 / expected_slope;

    assert_relative_eq!(fit.decay_length, expected_lambda, epsilon = 1e-2);
    assert_eq!(fit.sign, 1);
}

#[test]
fn fit_noisy_curve_r_squared_lower() {
    let clean = synth_exp(0.8, 50.0);
    let noise = [0.05_f32, -0.03, 0.02, -0.04, 0.01];
    let scores: Vec<f32> = clean.iter().zip(noise.iter()).map(|(c, n)| c + n).collect();

    let fit = fit_exponential_decay(&BANDWIDTHS, &scores);
    assert!(
        (fit.decay_length - 50.0).abs() < 10.0,
        "expected λ ≈ 50, got {}",
        fit.decay_length
    );
    assert!(
        fit.r_squared < 0.999 && fit.r_squared > 0.7,
        "expected 0.7 < r² < 0.999, got {}",
        fit.r_squared
    );
}
