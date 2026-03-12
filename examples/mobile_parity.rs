//! Mobile DNG parity: compare our dt_sigmoid pipeline against darktable's
//! default scene-referred rendering on mobile (iPhone ProRAW, Samsung Galaxy) DNGs.
//!
//! Unlike the FiveK dataset, there are no expert edits — we compare directly
//! against darktable's sigmoid output.
//!
//! Performance: all optimization runs on downscaled (~512px) data. Full-res
//! is only used for final image output. Uses zenresize (SIMD) and zencodecs
//! instead of the image crate.
//!
//! Usage: cargo run --release --features experimental --example mobile_parity
//!
//! Requires darktable-cli in PATH.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use imgref::ImgVec;
use rgb::Rgb;
use zencodecs::{DecodeRequest, EncodeRequest, ImageFormat};
use zenfilters::filters::cat16;
use zenfilters::filters::dt_sigmoid;
use zenfilters::regional::{RegionalComparison, RegionalFeatures};
use zenfilters::{OklabPlanes, scatter_srgb_u8_to_oklab};
use zenpixels::ColorPrimaries;
use zenpixels_convert::gamut::GamutMatrix;
use zenpixels_convert::oklab;
use zenraw::darktable::{self, DtConfig};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
use zensim::{RgbSlice, Zensim, ZensimProfile};

const OUTPUT_DIR: &str = "/mnt/v/output/zenfilters/mobile_parity";
const MAX_DIM: u32 = 512;

/// Mobile DNG files to test.
const MOBILE_DNGS: &[(&str, &str)] = &[
    ("iPhone_3269", "/mnt/v/heic/IMG_3269.DNG"),
    ("iPhone_3270", "/mnt/v/heic/IMG_3270.DNG"),
    ("iPhone_46CD", "/mnt/v/heic/46CD6167-C36B-4F98-B386-2300D8E840F0.DNG"),
    ("iPhone_CBFA", "/mnt/v/heic/CBFA569A-5C28-468E-96B4-CFFBAEB951C7.DNG"),
    ("Samsung_Fold7", "/mnt/v/heic/android/20260220_093521.dng"),
];

fn main() {
    fs::create_dir_all(OUTPUT_DIR).unwrap();

    if !darktable::is_available() {
        eprintln!("ERROR: darktable-cli not found in PATH");
        std::process::exit(1);
    }
    println!("darktable: {}", darktable::version().unwrap_or_default());

    let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
    let zs = Zensim::new(ZensimProfile::latest());
    let dt_config = DtConfig::new();

    let mut results: Vec<MobileResult> = Vec::new();

    for (label, path) in MOBILE_DNGS {
        let dng_path = PathBuf::from(path);
        if !dng_path.exists() {
            println!("\n{label}: SKIP (file not found)");
            continue;
        }
        let file_size = fs::metadata(&dng_path).map(|m| m.len()).unwrap_or(0);
        let t0 = Instant::now();
        println!("\n=== {label} ({:.1} MB) ===", file_size as f64 / 1_048_576.0);

        // 1. Read EXIF metadata
        let dng_bytes = fs::read(&dng_path).unwrap();
        let exif = zenraw::exif::read_metadata(&dng_bytes);
        if let Some(ref e) = exif {
            println!("  Make:     {:?}", e.make);
            println!("  Model:    {:?}", e.model);
            println!("  ISO:      {:?}", e.iso);
            println!("  Dims:     {:?}x{:?}", e.width, e.height);
            println!("  DNG ver:  {:?}", e.dng_version);
            println!("  BaseLine: {:?} EV", e.baseline_exposure);
            println!("  AsShotN:  {:?}", e.as_shot_neutral);
            println!("  WhiteXY:  {:?}", e.as_shot_white_xy);
            println!("  CM1 len:  {:?}", e.color_matrix_1.as_ref().map(|v| v.len()));
            println!("  CM2 len:  {:?}", e.color_matrix_2.as_ref().map(|v| v.len()));
            println!("  Illum1/2: {:?}/{:?}", e.calibration_illuminant_1, e.calibration_illuminant_2);
        } else {
            println!("  EXIF: failed to read");
        }

        // 2. Extract illuminant xy
        let illuminant_xy = exif.as_ref().and_then(|e| {
            let cm = if e.calibration_illuminant_2 == Some(21) {
                e.color_matrix_2.as_deref().or(e.color_matrix_1.as_deref())
            } else {
                e.color_matrix_1.as_deref()
            };
            cat16::illuminant_xy_from_dng(
                e.as_shot_white_xy,
                e.as_shot_neutral.as_deref(),
                cm,
            )
        });
        if let Some((x, y)) = illuminant_xy {
            println!("  Illuminant: ({x:.4}, {y:.4})");
        }

        // 3. Render with darktable (scene-referred sigmoid = default) → PNG
        let dt_sigmoid = darktable_render_png(&dng_path, "scene-referred (sigmoid)");
        if dt_sigmoid.is_none() {
            println!("  darktable sigmoid render FAILED");
            continue;
        }
        let (dt_sig_out, dtw, dth) = dt_sigmoid.unwrap();
        println!("  darktable sigmoid: {}x{} ({:.1}s)", dtw, dth, t0.elapsed().as_secs_f32());

        // 4. Render with darktable (workflow=none for linear)
        let linear_output = darktable::decode_file(&dng_path, &dt_config);
        if linear_output.is_err() {
            println!("  darktable linear render FAILED: {:?}", linear_output.err());
            continue;
        }
        let output = linear_output.unwrap();
        let pixels = output.pixels;
        let dw = pixels.width();
        let dh = pixels.height();
        let raw_bytes = pixels.copy_to_contiguous_bytes();
        let linear_f32: &[f32] = bytemuck::cast_slice(&raw_bytes);
        println!("  darktable linear: {}x{} ({:.1}s)", dw, dh, t0.elapsed().as_secs_f32());

        // 5. Analyze linear data range (on full-res, cheap)
        let n = linear_f32.len();
        let npix = n / 3;
        let (mut min_v, mut max_v) = (f32::MAX, f32::MIN);
        let (mut mr, mut mg, mut mb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..npix {
            let r = linear_f32[i * 3];
            let g = linear_f32[i * 3 + 1];
            let b = linear_f32[i * 3 + 2];
            min_v = min_v.min(r.min(g).min(b));
            max_v = max_v.max(r.max(g).max(b));
            mr += r as f64;
            mg += g as f64;
            mb += b as f64;
        }
        mr /= npix as f64;
        mg /= npix as f64;
        mb /= npix as f64;
        let mean_v = (mr + mg + mb) / 3.0;
        let clipped = linear_f32.iter().filter(|&&v| v > 0.95).count();
        let clipped_pct = 100.0 * clipped as f64 / n as f64;
        println!("  Linear range: [{min_v:.4}, {max_v:.4}] mean={mean_v:.4} clipped={clipped_pct:.2}%");
        println!("  Channel means: R={mr:.4} G={mg:.4} B={mb:.4}  R/G={:.3} B/G={:.3}",
            mr / mg, mb / mg);

        // 6. DOWNSCALE both linear and reference to MAX_DIM for optimization
        let (small_linear, sw, sh) = downscale_linear_f32(linear_f32, dw, dh, MAX_DIM);
        let (small_ref, srw, srh) = downscale_rgb8(&dt_sig_out, dtw, dth, MAX_DIM);
        // Crop to common dimensions
        let cw = sw.min(srw);
        let ch = sh.min(srh);
        let small_linear = crop_f32(&small_linear, sw, sh, cw, ch);
        let small_ref = crop_u8(&small_ref, srw, srh, cw, ch);
        println!("  Optimization res: {}x{} ({:.1}s)", cw, ch, t0.elapsed().as_secs_f32());

        // 7. Optimize uniform exposure multiplier (on downscaled data)
        let (optimal_mult, parity_uniform) = optimize_dt_sigmoid_exposure(
            &small_linear, cw, ch,
            &small_ref, cw, ch,
            &zs, 0.5, 8.0,
        );
        println!("  Uniform mult: {optimal_mult:.3}x → parity={parity_uniform:.1} ({:.1}s)",
            t0.elapsed().as_secs_f32());

        // 8. Optimize per-channel RGB exposure (on downscaled data)
        let (rgb_mult, parity_rgb) = optimize_rgb_exposure(
            &small_linear, cw, ch,
            &small_ref, cw, ch,
            &zs, optimal_mult,
        );
        let delta = parity_rgb - parity_uniform;
        println!("  RGB mult: [{:.3}, {:.3}, {:.3}] → parity={parity_rgb:.1} (Δ={delta:+.1}) ({:.1}s)",
            rgb_mult[0], rgb_mult[1], rgb_mult[2], t0.elapsed().as_secs_f32());
        println!("  RGB ratios: R/G={:.3}  B/G={:.3}",
            rgb_mult[0] / rgb_mult[1], rgb_mult[2] / rgb_mult[1]);

        // 9. Regional comparison on downscaled data
        let best_small = apply_dt_sigmoid_rgb(&small_linear, cw, ch, rgb_mult);
        let regional = regional_compare_srgb(&best_small, &small_ref, cw, ch, &m1);
        print_regional(&regional);

        // 10. Save comparison images (from downscaled, fast)
        let prefix = format!("{OUTPUT_DIR}/{label}");
        save_rgb8_jpeg(&small_ref, cw, ch, &format!("{prefix}_dt_sigmoid.jpg"));
        save_rgb8_jpeg(&best_small, cw, ch, &format!("{prefix}_our_rgb.jpg"));
        let (sbs, sbs_w, sbs_h) = side_by_side(&best_small, &small_ref, cw, ch);
        save_rgb8_jpeg(&sbs, sbs_w, sbs_h, &format!("{prefix}_sbs.jpg"));
        let heat = diff_heatmap(&best_small, &small_ref, cw, ch);
        save_rgb8_jpeg(&heat, cw, ch, &format!("{prefix}_diff.jpg"));

        // 11. Also render darktable basecurve for comparison
        let dt_basecurve = darktable_render_png(&dng_path, "display-referred");
        let sig_vs_bc = dt_basecurve.as_ref().map(|(bc_out, bw, bh)| {
            let (a, b, w, h) = resize_pair_rgb8(&dt_sig_out, dtw, dth, bc_out, *bw, *bh);
            zensim_score(&a, &b, w, h, &zs)
        });
        if let Some(s) = sig_vs_bc {
            println!("  dt sigmoid vs basecurve: {s:.1}");
        }

        println!("  Total: {:.1}s", t0.elapsed().as_secs_f32());

        results.push(MobileResult {
            name: label.to_string(),
            parity_uniform,
            parity_rgb,
            optimal_mult,
            rgb_mult,
            illuminant_xy,
            make: exif.as_ref().and_then(|e| e.make.clone()),
            model: exif.as_ref().and_then(|e| e.model.clone()),
            iso: exif.as_ref().and_then(|e| e.iso),
            baseline_exposure: exif.as_ref().and_then(|e| e.baseline_exposure),
            dims: (dw, dh),
            linear_mean: mean_v as f32,
            regional,
        });
    }

    // Summary table
    println!("\n\n=== MOBILE DNG PARITY SUMMARY ===");
    println!(
        "{:<20} {:>8} {:>8} {:>8} {:>25} {:>8} {:>10}",
        "Name", "Uniform", "RGB", "Δ", "R,G,B mults", "Mult", "BL Exp"
    );
    println!("{}", "-".repeat(100));

    for r in &results {
        let delta = r.parity_rgb - r.parity_uniform;
        let rgb_str = format!("{:.3},{:.3},{:.3}", r.rgb_mult[0], r.rgb_mult[1], r.rgb_mult[2]);
        let bl = r.baseline_exposure.map_or("---".to_string(), |v| format!("{v:+.2}"));
        println!(
            "{:<20} {:>8.1} {:>8.1} {:>+8.1} {:>25} {:>8.3} {:>10}",
            r.name, r.parity_uniform, r.parity_rgb, delta, rgb_str, r.optimal_mult, bl,
        );
    }

    if !results.is_empty() {
        println!("{}", "-".repeat(100));
        let n = results.len() as f64;
        let mean_u: f64 = results.iter().map(|r| r.parity_uniform).sum::<f64>() / n;
        let mean_rgb: f64 = results.iter().map(|r| r.parity_rgb).sum::<f64>() / n;
        let mean_mult: f32 = results.iter().map(|r| r.optimal_mult).sum::<f32>() / n as f32;
        println!(
            "{:<20} {:>8.1} {:>8.1} {:>+8.1} {:>25} {:>8.3}",
            "MEAN", mean_u, mean_rgb, mean_rgb - mean_u, "", mean_mult,
        );

        println!("\n--- Exposure multiplier comparison ---");
        println!("  DSLR FiveK mean: ~1.85x (from prior runs)");
        println!("  Mobile mean:     {mean_mult:.3}x");
        let mult_range: (f32, f32) = results.iter().fold(
            (f32::MAX, f32::MIN),
            |(lo, hi), r| (lo.min(r.optimal_mult), hi.max(r.optimal_mult)),
        );
        println!("  Mobile range:    [{:.3}, {:.3}]", mult_range.0, mult_range.1);
    }

    // Save TSV
    let tsv_path = format!("{OUTPUT_DIR}/mobile_parity.tsv");
    let mut tsv = String::new();
    tsv.push_str("name\tmake\tmodel\tiso\tbaseline_exp\tdims\tlinear_mean\tparity_uniform\tparity_rgb\topt_mult\trgb_r\trgb_g\trgb_b\tilluminant_x\tilluminant_y\tregional_agg\n");
    for r in &results {
        let (ix, iy) = r.illuminant_xy.unwrap_or((-1.0, -1.0));
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{:.2}\t{}x{}\t{:.4}\t{:.2}\t{:.2}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.4}\t{:.4}\t{:.4}\n",
            r.name,
            r.make.as_deref().unwrap_or("?"),
            r.model.as_deref().unwrap_or("?"),
            r.iso.unwrap_or(0),
            r.baseline_exposure.unwrap_or(0.0),
            r.dims.0, r.dims.1,
            r.linear_mean,
            r.parity_uniform, r.parity_rgb,
            r.optimal_mult, r.rgb_mult[0], r.rgb_mult[1], r.rgb_mult[2],
            ix, iy,
            r.regional.aggregate,
        ));
    }
    fs::write(&tsv_path, &tsv).unwrap();
    println!("\nResults saved to {tsv_path}");
}

struct MobileResult {
    name: String,
    parity_uniform: f64,
    parity_rgb: f64,
    optimal_mult: f32,
    rgb_mult: [f32; 3],
    illuminant_xy: Option<(f32, f32)>,
    make: Option<String>,
    model: Option<String>,
    iso: Option<u32>,
    baseline_exposure: Option<f64>,
    dims: (u32, u32),
    linear_mean: f32,
    regional: RegionalComparison,
}

// ── Downscaling (zenresize) ─────────────────────────────────────────────

/// Downscale linear f32 RGB data to fit within max_dim using zenresize (SIMD).
fn downscale_linear_f32(data: &[f32], w: u32, h: u32, max_dim: u32) -> (Vec<f32>, u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (data.to_vec(), w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    let nw = ((w as f64 * scale) as u32).max(1);
    let nh = ((h as f64 * scale) as u32).max(1);

    let config = ResizeConfig::builder(w, h, nw, nh)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBF32_LINEAR)
        .build();
    let mut resizer = Resizer::new(&config);
    let out = resizer.resize_f32(data);
    (out, nw, nh)
}

/// Downscale sRGB u8 RGB data to fit within max_dim using zenresize (SIMD).
fn downscale_rgb8(data: &[u8], w: u32, h: u32, max_dim: u32) -> (Vec<u8>, u32, u32) {
    if w <= max_dim && h <= max_dim {
        return (data.to_vec(), w, h);
    }
    let scale = max_dim as f64 / w.max(h) as f64;
    let nw = ((w as f64 * scale) as u32).max(1);
    let nh = ((h as f64 * scale) as u32).max(1);

    let config = ResizeConfig::builder(w, h, nw, nh)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGB8_SRGB)
        .build();
    let mut resizer = Resizer::new(&config);
    let out = resizer.resize(data);
    (out, nw, nh)
}

/// Crop f32 RGB data to target dimensions (top-left).
fn crop_f32(data: &[f32], w: u32, h: u32, tw: u32, th: u32) -> Vec<f32> {
    if tw == w && th == h { return data.to_vec(); }
    let tw = tw.min(w);
    let th = th.min(h);
    let mut out = vec![0.0f32; (tw as usize) * (th as usize) * 3];
    for y in 0..th as usize {
        let src_off = y * (w as usize) * 3;
        let dst_off = y * (tw as usize) * 3;
        let row_bytes = (tw as usize) * 3;
        out[dst_off..dst_off + row_bytes].copy_from_slice(&data[src_off..src_off + row_bytes]);
    }
    out
}

/// Crop u8 RGB data to target dimensions (top-left).
fn crop_u8(data: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    if tw == w && th == h { return data.to_vec(); }
    let tw = tw.min(w);
    let th = th.min(h);
    let mut out = vec![0u8; (tw as usize) * (th as usize) * 3];
    for y in 0..th as usize {
        let src_off = y * (w as usize) * 3;
        let dst_off = y * (tw as usize) * 3;
        let row_bytes = (tw as usize) * 3;
        out[dst_off..dst_off + row_bytes].copy_from_slice(&data[src_off..src_off + row_bytes]);
    }
    out
}

// ── Darktable rendering ─────────────────────────────────────────────────

/// Render a DNG through darktable-cli, outputting PNG. Returns RGB8 pixel data.
fn darktable_render_png(dng_path: &Path, workflow: &str) -> Option<(Vec<u8>, u32, u32)> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_dir = PathBuf::from(format!("/tmp/dt_mobile_{}_{}", std::process::id(), id));
    fs::create_dir_all(&tmp_dir).ok()?;
    let out_path = tmp_dir.join("output.png");

    let status = Command::new("darktable-cli")
        .arg(dng_path)
        .arg(&out_path)
        .arg("--icc-type")
        .arg("SRGB")
        .arg("--apply-custom-presets")
        .arg("false")
        .arg("--core")
        .arg("--library")
        .arg(":memory:")
        .arg("--configdir")
        .arg(tmp_dir.join("dtconf"))
        .arg("--conf")
        .arg(format!("plugins/darkroom/workflow={workflow}"))
        .stderr(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .status()
        .ok()?;

    if !status.success() {
        let _ = fs::remove_dir_all(&tmp_dir);
        return None;
    }

    let png_bytes = fs::read(&out_path).ok()?;
    let _ = fs::remove_dir_all(&tmp_dir);

    // Decode PNG with zencodecs
    let decoded = DecodeRequest::new(&png_bytes).decode().ok()?;
    let w = decoded.width();
    let h = decoded.height();
    // Convert to RGB8 contiguous bytes
    use zenpixels_convert::PixelBufferConvertTypedExt;
    let rgb8_buf = decoded.into_buffer().to_rgb8();
    let bytes = rgb8_buf.copy_to_contiguous_bytes();
    Some((bytes, w, h))
}

// ── dt_sigmoid pipeline ─────────────────────────────────────────────────

fn apply_dt_sigmoid_rgb(linear_f32: &[f32], _w: u32, _h: u32, rgb_mult: [f32; 3]) -> Vec<u8> {
    let params = dt_sigmoid::default_params();
    let mut rgb = linear_f32.to_vec();
    let n = rgb.len() / 3;
    for i in 0..n {
        let base = i * 3;
        rgb[base] *= rgb_mult[0];
        rgb[base + 1] *= rgb_mult[1];
        rgb[base + 2] *= rgb_mult[2];
    }
    dt_sigmoid::apply_dt_sigmoid(&mut rgb, &params);
    linear_to_srgb_u8(&rgb)
}

fn apply_dt_sigmoid_uniform(linear_f32: &[f32], _w: u32, _h: u32, mult: f32) -> Vec<u8> {
    let params = dt_sigmoid::default_params();
    let mut rgb = linear_f32.to_vec();
    if (mult - 1.0).abs() > 1e-6 {
        for v in rgb.iter_mut() {
            *v *= mult;
        }
    }
    dt_sigmoid::apply_dt_sigmoid(&mut rgb, &params);
    linear_to_srgb_u8(&rgb)
}

fn linear_to_srgb_u8(rgb: &[f32]) -> Vec<u8> {
    let mut output = vec![0u8; rgb.len()];
    for (i, &v) in rgb.iter().enumerate() {
        let v = v.clamp(0.0, 1.0);
        let srgb = if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        };
        output[i] = (srgb * 255.0 + 0.5) as u8;
    }
    output
}

// ── Optimization ────────────────────────────────────────────────────────

fn golden_search(f: impl Fn(f32) -> f64, lo: f32, hi: f32) -> (f32, f64) {
    let phi = (5.0f32.sqrt() - 1.0) / 2.0;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..25 {
        if fc > fd {
            b = d; d = c; fd = fc;
            c = b - phi * (b - a);
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + phi * (b - a);
            fd = f(d);
        }
        if (b - a).abs() < 0.005 { break; }
    }
    let best = (a + b) / 2.0;
    (best, f(best))
}

fn optimize_dt_sigmoid_exposure(
    linear_f32: &[f32], w: u32, h: u32,
    reference: &[u8], _rw: u32, _rh: u32,
    zs: &Zensim, lo: f32, hi: f32,
) -> (f32, f64) {
    golden_search(
        |mult| {
            let out = apply_dt_sigmoid_uniform(linear_f32, w, h, mult);
            zensim_score(&out, reference, w, h, zs)
        },
        lo, hi,
    )
}

fn optimize_rgb_exposure(
    linear_f32: &[f32], w: u32, h: u32,
    reference: &[u8], _rw: u32, _rh: u32,
    zs: &Zensim, uniform_mult: f32,
) -> ([f32; 3], f64) {
    let mut rgb = [uniform_mult; 3];
    let mut best_score = 0.0f64;

    let eval = |m: [f32; 3]| -> f64 {
        let out = apply_dt_sigmoid_rgb(linear_f32, w, h, m);
        zensim_score(&out, reference, w, h, zs)
    };

    for _ in 0..3 {
        for ch in 0..3 {
            let (best_val, score) = golden_search(
                |v| {
                    let mut m = rgb;
                    m[ch] = v;
                    eval(m)
                },
                (rgb[ch] * 0.5).max(0.2), rgb[ch] * 2.0,
            );
            rgb[ch] = best_val;
            best_score = score;
        }
    }

    (rgb, best_score)
}

// ── Resize & comparison helpers ─────────────────────────────────────────

/// Resize two RGB8 images to common dimensions <= MAX_DIM.
fn resize_pair_rgb8(
    a: &[u8], aw: u32, ah: u32,
    b: &[u8], bw: u32, bh: u32,
) -> (Vec<u8>, Vec<u8>, u32, u32) {
    let (ra, raw, rah) = downscale_rgb8(a, aw, ah, MAX_DIM);
    let (rb, rbw, rbh) = downscale_rgb8(b, bw, bh, MAX_DIM);
    let w = raw.min(rbw);
    let h = rah.min(rbh);
    let ca = crop_u8(&ra, raw, rah, w, h);
    let cb = crop_u8(&rb, rbw, rbh, w, h);
    (ca, cb, w, h)
}

fn zensim_score(a: &[u8], b: &[u8], w: u32, h: u32, zs: &Zensim) -> f64 {
    let expected = w as usize * h as usize * 3;
    if a.len() != expected || b.len() != expected {
        eprintln!("    zensim: buffer mismatch: a={} b={} expected={} ({}x{})",
            a.len(), b.len(), expected, w, h);
        return 0.0;
    }
    let a_rgb: &[[u8; 3]] = bytemuck::cast_slice(a);
    let b_rgb: &[[u8; 3]] = bytemuck::cast_slice(b);
    let sa = RgbSlice::new(a_rgb, w as usize, h as usize);
    let sb = RgbSlice::new(b_rgb, w as usize, h as usize);
    match zs.compute(&sa, &sb) {
        Ok(r) => r.score(),
        Err(e) => {
            eprintln!("    zensim error: {e}");
            0.0
        }
    }
}

fn save_rgb8_jpeg(data: &[u8], w: u32, h: u32, path: &str) {
    let pixels: &[Rgb<u8>] = bytemuck::cast_slice(data);
    let img = ImgVec::new(pixels.to_vec(), w as usize, h as usize);
    match EncodeRequest::new(ImageFormat::Jpeg)
        .with_quality(90.0)
        .encode_rgb8(img.as_ref())
    {
        Ok(encoded) => { let _ = fs::write(path, encoded.data()); }
        Err(e) => eprintln!("    save error: {e}"),
    }
}

fn diff_heatmap(a: &[u8], b: &[u8], w: u32, h: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        let idx = i * 3;
        let dr = (a[idx] as i32 - b[idx] as i32).unsigned_abs();
        let dg = (a[idx + 1] as i32 - b[idx + 1] as i32).unsigned_abs();
        let db = (a[idx + 2] as i32 - b[idx + 2] as i32).unsigned_abs();
        let d = dr.max(dg).max(db).min(255) as u8;
        let v = (d as u32 * 4).min(255) as u8;
        if v < 128 {
            out[idx] = 0;
            out[idx + 1] = v;
            out[idx + 2] = v * 2;
        } else {
            let t = v - 128;
            out[idx] = 128 + t;
            out[idx + 1] = 128 - t;
            out[idx + 2] = t;
        }
    }
    out
}

fn side_by_side(a: &[u8], b: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let heat = diff_heatmap(a, b, w, h);
    let total_w = w * 3;
    let stride_a = (w as usize) * 3;
    let stride_out = (total_w as usize) * 3;
    let mut out = vec![0u8; stride_out * (h as usize)];
    for y in 0..h as usize {
        let row_a = &a[y * stride_a..(y + 1) * stride_a];
        let row_b = &b[y * stride_a..(y + 1) * stride_a];
        let row_h = &heat[y * stride_a..(y + 1) * stride_a];
        let row_out = &mut out[y * stride_out..(y + 1) * stride_out];
        row_out[..stride_a].copy_from_slice(row_a);
        row_out[stride_a..stride_a * 2].copy_from_slice(row_b);
        row_out[stride_a * 2..stride_a * 3].copy_from_slice(row_h);
    }
    (out, total_w, h)
}

fn regional_compare_srgb(
    a: &[u8], b: &[u8], w: u32, h: u32, m1: &GamutMatrix,
) -> RegionalComparison {
    let mut planes_a = OklabPlanes::new(w, h);
    scatter_srgb_u8_to_oklab(a, &mut planes_a, 3, m1);
    let mut planes_b = OklabPlanes::new(w, h);
    scatter_srgb_u8_to_oklab(b, &mut planes_b, 3, m1);
    let fa = RegionalFeatures::extract(&planes_a);
    let fb = RegionalFeatures::extract(&planes_b);
    RegionalComparison::compare(&fa, &fb)
}

fn print_regional(r: &RegionalComparison) {
    let labels = RegionalComparison::zone_labels();
    let lum: Vec<String> = labels.luminance.iter().zip(r.lum_zone_dist.iter())
        .map(|(l, v)| format!("{l}={v:.3}")).collect();
    let hue: Vec<String> = labels.hue.iter().zip(r.hue_sector_dist.iter())
        .map(|(l, v)| format!("{l}={v:.3}")).collect();
    let chr: Vec<String> = labels.chroma.iter().zip(r.chroma_zone_dist.iter())
        .map(|(l, v)| format!("{l}={v:.3}")).collect();
    println!("  Regional L: {}", lum.join("  "));
    println!("  Regional H: {}", hue.join("  "));
    println!("  Regional C: {}", chr.join("  "));
    println!("  Regional aggregate: {:.4}", r.aggregate);
}
