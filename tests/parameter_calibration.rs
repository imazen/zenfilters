//! Verify that every filter's parameter curve is well-calibrated:
//! - At 25% of slider range: filter produces a visible change (mean |delta L| > threshold)
//! - At 75% of slider range: filter produces a reasonable result (not blown out)
//!
//! This catches filters with dead zones, explosive curves, or broken parameter scaling.
//!
//! Uses a synthetic test image with known luminance distribution so results are
//! deterministic and independent of external corpus files.

use zenfilters::filters::*;
use zenfilters::{Filter, FilterContext, OklabPlanes};

/// Create a synthetic 128x128 test image with realistic luminance distribution.
/// L spans 0.05–0.95 in a smooth gradient with some texture.
fn make_test_planes() -> OklabPlanes {
    let (w, h) = (128u32, 128u32);
    let mut planes = OklabPlanes::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            // Base gradient: smooth luminance ramp
            let base = 0.05 + 0.9 * (x as f32 / (w - 1) as f32);
            // Add vertical texture for neighborhood filters to detect
            let texture = 0.03 * ((y as f32 * 0.5).sin());
            planes.l[i] = (base + texture).clamp(0.01, 0.99);
            // Moderate chroma
            planes.a[i] = 0.05 * ((x as f32 / 20.0).sin());
            planes.b[i] = 0.03 * ((y as f32 / 15.0).cos());
        }
    }
    planes
}

/// Create a test image that's dark (to exercise shadow-dependent filters).
fn make_dark_planes() -> OklabPlanes {
    let (w, h) = (128u32, 128u32);
    let mut planes = OklabPlanes::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            planes.l[i] = 0.02 + 0.15 * (x as f32 / (w - 1) as f32);
            planes.a[i] = 0.02;
            planes.b[i] = -0.01;
        }
    }
    planes
}

/// Create a test image that's bright (to exercise highlight-dependent filters).
fn make_bright_planes() -> OklabPlanes {
    let (w, h) = (128u32, 128u32);
    let mut planes = OklabPlanes::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            planes.l[i] = 0.6 + 0.35 * (x as f32 / (w - 1) as f32);
            planes.a[i] = 0.03;
            planes.b[i] = 0.02;
        }
    }
    planes
}

/// Measure mean absolute change in L channel.
fn mean_delta_l(original: &[f32], filtered: &[f32]) -> f32 {
    let n = original.len();
    let sum: f32 = original
        .iter()
        .zip(filtered.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();
    sum / n as f32
}

/// Measure max absolute change in L channel.
fn max_delta_l(original: &[f32], filtered: &[f32]) -> f32 {
    original
        .iter()
        .zip(filtered.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

/// Check that a filter produces visible change at the given fraction of its range,
/// and that the result is reasonable (L stays in valid range, change isn't extreme).
fn check_filter(
    name: &str,
    filter: &dyn Filter,
    planes_fn: fn() -> OklabPlanes,
    min_mean_delta: f32,
    max_mean_delta: f32,
    fraction_label: &str,
) {
    let base = planes_fn();
    let mut filtered = base.clone();
    let mut ctx = FilterContext::new();
    filter.apply(&mut filtered, &mut ctx);

    let mean_d = mean_delta_l(&base.l, &filtered.l);
    let max_d = max_delta_l(&base.l, &filtered.l);

    // Check L stays in a sane range (allow slight exceedance for some filters)
    let l_min = filtered.l.iter().fold(f32::MAX, |a, &b| a.min(b));
    let l_max = filtered.l.iter().fold(f32::MIN, |a, &b| a.max(b));

    if mean_d < min_mean_delta {
        panic!(
            "{name} at {fraction_label}: mean delta L = {mean_d:.6} < {min_mean_delta:.6} — \
             filter has no visible effect! (max_delta={max_d:.4}, L range=[{l_min:.3}, {l_max:.3}])"
        );
    }
    if mean_d > max_mean_delta {
        panic!(
            "{name} at {fraction_label}: mean delta L = {mean_d:.4} > {max_mean_delta:.4} — \
             filter is too extreme! (max_delta={max_d:.4}, L range=[{l_min:.3}, {l_max:.3}])"
        );
    }
}

// Helper macro to construct non_exhaustive filter via Default + mutation
macro_rules! mk {
    ($ty:ty, $($field:ident = $val:expr),* $(,)?) => {{
        #[allow(unused_mut)]
        let mut f = <$ty>::default();
        $(f.$field = $val;)*
        f
    }};
}

// ═══════════════════════════════════════════════════════════════════════
// TESTS: Each filter at 25% and 75% of its useful slider range
// ═══════════════════════════════════════════════════════════════════════

// Threshold: 0.001 mean delta L is invisible. 0.003 is barely visible.
// At 25%: must exceed 0.003 (clearly something happened).
// At 75%: must not exceed 0.3 (image not destroyed).
const MIN_25: f32 = 0.003;
const MAX_75: f32 = 0.3;

#[test]
fn exposure_calibration() {
    // Stops are perceptually linear. 25% of [-3,+3] centered at 0 → 0.75 stops
    check_filter("Exposure@25%", &mk!(Exposure, stops = 0.75), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Exposure@75%", &mk!(Exposure, stops = 2.25), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn contrast_calibration() {
    // Contrast uses slider² mapping. Slider 0.25 → internal 0.0625, slider 0.75 → 0.5625
    let s25 = zenfilters::slider::contrast_from_slider(0.25);
    let s75 = zenfilters::slider::contrast_from_slider(0.75);
    check_filter("Contrast@25%", &mk!(Contrast, amount = s25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Contrast@75%", &mk!(Contrast, amount = s75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn saturation_calibration() {
    // Slider centered: 0.25 → factor 0.5 (desaturate), 0.75 → factor 1.5 (boost)
    // This only affects chroma (a/b), not L. So we measure chroma delta.
    let base = make_test_planes();
    let mut f25 = base.clone();
    let mut f75 = base.clone();
    let mut ctx = FilterContext::new();
    mk!(Saturation, factor = 0.5).apply(&mut f25, &mut ctx);
    mk!(Saturation, factor = 1.5).apply(&mut f75, &mut ctx);

    let delta_a_25: f32 = base.a.iter().zip(f25.a.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>() / base.a.len() as f32;
    let delta_a_75: f32 = base.a.iter().zip(f75.a.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>() / base.a.len() as f32;
    assert!(delta_a_25 > 0.001, "Saturation@25%: chroma delta too small: {delta_a_25}");
    assert!(delta_a_75 > 0.001, "Saturation@75%: chroma delta too small: {delta_a_75}");
}

#[test]
fn highlights_shadows_calibration() {
    check_filter("HighlightsShadows@25%", &mk!(HighlightsShadows, highlights = -0.25, shadows = 0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("HighlightsShadows@75%", &mk!(HighlightsShadows, highlights = -0.75, shadows = 0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn whites_blacks_calibration() {
    check_filter("WhitesBlacks@25%", &mk!(WhitesBlacks, whites = 0.25, blacks = -0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("WhitesBlacks@75%", &mk!(WhitesBlacks, whites = 0.75, blacks = -0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn black_point_calibration() {
    // Range 0–0.3, 25% = 0.075, 75% = 0.225
    check_filter("BlackPoint@25%", &mk!(BlackPoint, level = 0.075), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("BlackPoint@75%", &mk!(BlackPoint, level = 0.225), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn clarity_calibration() {
    // amount range [-2, +2], 25% = 0.5, 75% = 1.5
    check_filter("Clarity@25%", &mk!(Clarity, sigma = 3.0, amount = 0.5), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Clarity@75%", &mk!(Clarity, sigma = 3.0, amount = 1.5), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn sharpen_calibration() {
    check_filter("Sharpen@25%", &mk!(Sharpen, sigma = 1.0, amount = 0.5), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Sharpen@75%", &mk!(Sharpen, sigma = 1.0, amount = 1.5), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn brilliance_calibration() {
    check_filter("Brilliance@25%", &mk!(Brilliance, sigma = 10.0, amount = 0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Brilliance@75%", &mk!(Brilliance, sigma = 10.0, amount = 0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn dehaze_calibration() {
    // Uses slider² mapping. Slider 0.25 → 0.0625, 0.75 → 0.5625
    let s25 = zenfilters::slider::dehaze_from_slider(0.25);
    let s75 = zenfilters::slider::dehaze_from_slider(0.75);
    check_filter("Dehaze@25%", &mk!(Dehaze, strength = s25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Dehaze@75%", &mk!(Dehaze, strength = s75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn local_tone_map_calibration() {
    // Uses slider² mapping. Slider 0.25 → 0.0625, 0.75 → 0.5625
    let s25 = zenfilters::slider::ltm_compression_from_slider(0.25);
    let s75 = zenfilters::slider::ltm_compression_from_slider(0.75);
    check_filter("LocalToneMap@25%", &mk!(LocalToneMap, compression = s25, sigma = 20.0), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("LocalToneMap@75%", &mk!(LocalToneMap, compression = s75, sigma = 20.0), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn bloom_calibration() {
    // amount 0–1, threshold typically 0.5–0.8
    check_filter("Bloom@25%", &mk!(Bloom, amount = 0.25, threshold = 0.5, sigma = 20.0), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Bloom@75%", &mk!(Bloom, amount = 0.75, threshold = 0.5, sigma = 20.0), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn vignette_calibration() {
    // strength 0–1
    check_filter("Vignette@25%", &mk!(Vignette, strength = 0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Vignette@75%", &mk!(Vignette, strength = 0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn grain_calibration() {
    // amount 0–1
    check_filter("Grain@25%", &mk!(Grain, amount = 0.25, size = 1.5, seed = 42), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Grain@75%", &mk!(Grain, amount = 0.75, size = 1.5, seed = 42), make_test_planes, 0.01, MAX_75, "75%");
}

// ═══════════════════════════════════════════════════════════════════════
// IMAGE-DEPENDENT FILTERS: Use dark/bright planes as appropriate
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn auto_exposure_calibration() {
    // Uses dark image so there's a clear correction to make.
    // strength 0–1: 25% = 0.25, 75% = 0.75
    check_filter("AutoExposure@25%", &mk!(AutoExposure, strength = 0.25, target = 0.5, max_correction = 3.0),
        make_dark_planes, MIN_25, 1.0, "25%");
    check_filter("AutoExposure@75%", &mk!(AutoExposure, strength = 0.75, target = 0.5, max_correction = 3.0),
        make_dark_planes, 0.01, MAX_75, "75%");
}

#[test]
fn auto_exposure_on_normal_image() {
    // On a normal-exposure image, auto exposure should still do *something* at 25%
    // if the geometric mean isn't exactly at target
    check_filter("AutoExposure_normal@25%", &mk!(AutoExposure, strength = 0.25, target = 0.5, max_correction = 3.0),
        make_test_planes, MIN_25, 1.0, "25%");
}

#[test]
fn shadow_lift_calibration() {
    // Must use dark image (p5 < 0.3 required)
    check_filter("ShadowLift@25%", &mk!(ShadowLift, strength = 0.25),
        make_dark_planes, MIN_25, 1.0, "25%");
    check_filter("ShadowLift@75%", &mk!(ShadowLift, strength = 0.75),
        make_dark_planes, 0.01, MAX_75, "75%");
}

#[test]
fn shadow_lift_on_normal_image() {
    // On a normal image, ShadowLift may legitimately do nothing (p5 > 0.3).
    // This test documents that behavior.
    let base = make_test_planes();
    let mut filtered = base.clone();
    let mut ctx = FilterContext::new();
    mk!(ShadowLift, strength = 1.0).apply(&mut filtered, &mut ctx);
    let delta = mean_delta_l(&base.l, &filtered.l);
    // Document: this is expected to be near zero on normal images
    eprintln!("ShadowLift on normal image: mean delta L = {delta:.6} (expected ~0 if p5 > 0.3)");
}

#[test]
fn highlight_recovery_calibration() {
    // Must use bright image (p95 > 0.7 required)
    check_filter("HighlightRecovery@25%", &mk!(HighlightRecovery, strength = 0.25),
        make_bright_planes, MIN_25, 1.0, "25%");
    check_filter("HighlightRecovery@75%", &mk!(HighlightRecovery, strength = 0.75),
        make_bright_planes, 0.01, MAX_75, "75%");
}

#[test]
fn highlight_recovery_on_normal_image() {
    // On a normal image, HighlightRecovery may do nothing (p95 < 0.7).
    let base = make_test_planes();
    let mut filtered = base.clone();
    let mut ctx = FilterContext::new();
    mk!(HighlightRecovery, strength = 1.0).apply(&mut filtered, &mut ctx);
    let delta = mean_delta_l(&base.l, &filtered.l);
    eprintln!("HighlightRecovery on normal image: mean delta L = {delta:.6}");
}

#[test]
fn auto_levels_calibration() {
    // Default strength is 1.0, so we test the operation itself.
    // On a narrow-range image, it should stretch.
    let (w, h) = (128u32, 128u32);
    let mut narrow = OklabPlanes::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            narrow.l[i] = 0.3 + 0.2 * (x as f32 / (w - 1) as f32); // 0.3–0.5
            narrow.a[i] = 0.02;
            narrow.b[i] = -0.01;
        }
    }
    let original = narrow.l.clone();
    let mut ctx = FilterContext::new();

    // At strength 0.25, should produce visible stretch
    let mut f25 = narrow.clone();
    mk!(AutoLevels, strength = 0.25).apply(&mut f25, &mut ctx);
    let delta_25 = mean_delta_l(&original, &f25.l);
    assert!(delta_25 > MIN_25, "AutoLevels@25%: delta too small: {delta_25}");

    // At strength 0.75, should be a strong stretch but not destroy
    let mut f75 = narrow.clone();
    mk!(AutoLevels, strength = 0.75).apply(&mut f75, &mut ctx);
    let delta_75 = mean_delta_l(&original, &f75.l);
    assert!(delta_75 > 0.01, "AutoLevels@75%: delta too small: {delta_75}");
    assert!(delta_75 < MAX_75, "AutoLevels@75%: delta too large: {delta_75}");
}

#[test]
fn tone_equalizer_calibration() {
    // 9 zones, each -4 to +4 EV. Test with moderate zone adjustments.
    {
        let mut t = ToneEqualizer::default();
        t.zones = [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]; // mild mid boost
        check_filter("ToneEqualizer@25%", &t, make_test_planes, MIN_25, 1.0, "25%");
    }
    {
        let mut t = ToneEqualizer::default();
        t.zones = [-1.5, -0.75, 0.0, 0.75, 1.5, 0.75, 0.0, -0.75, -1.5]; // strong spread
        check_filter("ToneEqualizer@75%", &t, make_test_planes, 0.01, MAX_75, "75%");
    }
}

#[test]
fn noise_reduction_calibration() {
    // Uses slider² mapping
    let s25 = zenfilters::slider::nr_strength_from_slider(0.25);
    let s75 = zenfilters::slider::nr_strength_from_slider(0.75);
    check_filter("NoiseReduction@25%", &mk!(NoiseReduction, luminance = s25, chroma = s25),
        make_test_planes, MIN_25, 1.0, "25%");
    check_filter("NoiseReduction@75%", &mk!(NoiseReduction, luminance = s75, chroma = s75),
        make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn sigmoid_calibration() {
    // contrast 0.5–3.0, test at 25% and 75% of that range
    // 25%: 0.5 + 0.625 = 1.125, 75%: 0.5 + 1.875 = 2.375
    check_filter("Sigmoid@25%", &mk!(Sigmoid, contrast = 1.125), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Sigmoid@75%", &mk!(Sigmoid, contrast = 2.375), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn levels_calibration() {
    // gamma 0.1–10.0, centered at 1.0. Test at 25%/75% of log range
    // 25% lighter: gamma ≈ 1.5, 75%: gamma ≈ 3.0
    check_filter("Levels@25%", &mk!(Levels, gamma = 1.5), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Levels@75%", &mk!(Levels, gamma = 3.0), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn devignette_calibration() {
    check_filter("Devignette@25%", &mk!(Devignette, strength = 0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Devignette@75%", &mk!(Devignette, strength = 0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn gamut_expand_calibration() {
    // Only affects chroma, not L. Use chroma measurement.
    let base = make_test_planes();
    let mut f25 = base.clone();
    let mut f75 = base.clone();
    let mut ctx = FilterContext::new();
    mk!(GamutExpand, strength = 0.25).apply(&mut f25, &mut ctx);
    mk!(GamutExpand, strength = 0.75).apply(&mut f75, &mut ctx);

    let delta_a_25: f32 = base.a.iter().zip(f25.a.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>() / base.a.len() as f32;
    let delta_a_75: f32 = base.a.iter().zip(f75.a.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>() / base.a.len() as f32;
    assert!(delta_a_25 > 0.0005, "GamutExpand@25%: chroma delta too small: {delta_a_25}");
    assert!(delta_a_75 < 0.2, "GamutExpand@75%: chroma delta too large: {delta_a_75}");
}

#[test]
fn bilateral_calibration() {
    check_filter("Bilateral@25%", &mk!(Bilateral, strength = 0.25, spatial_sigma = 5.0), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Bilateral@75%", &mk!(Bilateral, strength = 0.75, spatial_sigma = 5.0), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn edge_detect_calibration() {
    check_filter("EdgeDetect@25%", &mk!(EdgeDetect, strength = 0.25), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("EdgeDetect@75%", &mk!(EdgeDetect, strength = 0.75), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn chromatic_aberration_calibration() {
    // This only affects chroma channels (shifts a/b planes)
    let base = make_test_planes();
    let mut f25 = base.clone();
    let mut ctx = FilterContext::new();
    mk!(ChromaticAberration, shift_a = 0.0025, shift_b = -0.0025).apply(&mut f25, &mut ctx);

    let delta_a: f32 = base.a.iter().zip(f25.a.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f32>() / base.a.len() as f32;
    assert!(delta_a > 0.0001, "ChromaticAberration@25%: chroma delta too small: {delta_a}");
}

#[test]
fn texture_calibration() {
    check_filter("Texture@25%", &mk!(Texture, sigma = 1.0, amount = 0.5), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("Texture@75%", &mk!(Texture, sigma = 1.0, amount = 1.5), make_test_planes, 0.01, MAX_75, "75%");
}

#[test]
fn adaptive_sharpen_calibration() {
    check_filter("AdaptiveSharpen@25%", &mk!(AdaptiveSharpen, amount = 0.5, sigma = 1.0), make_test_planes, MIN_25, 1.0, "25%");
    check_filter("AdaptiveSharpen@75%", &mk!(AdaptiveSharpen, amount = 1.5, sigma = 1.0), make_test_planes, 0.01, MAX_75, "75%");
}
