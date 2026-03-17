//! Benchmark: planar pipeline (scatter→adjust→gather) vs fused interleaved.
//!
//! Tests per-pixel-only filter chains at 1080p and 4K.
//! The fused path does RGB→Oklab→adjust→RGB in one streaming SIMD pass.
//!
//! Run: `just fused-bench`

use std::sync::Arc;
use zenbench::{Suite, Throughput};
use zenfilters::{FilterContext, Pipeline, PipelineConfig};
use zenpixels::ColorPrimaries;
use zenpixels_convert::oklab;

fn make_rgb_data(w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut data = vec![0.0f32; n * 3];
    for i in 0..n {
        let t = i as f32 / n as f32;
        data[i * 3] = t * 0.8 + 0.1;
        data[i * 3 + 1] = (1.0 - t) * 0.7 + 0.15;
        data[i * 3 + 2] = t * 0.5 + 0.2;
    }
    data
}

fn bench_fused_group(suite: &mut Suite, w: u32, h: u32, label: &str) {
    let n = (w as usize) * (h as usize);
    let src = Arc::new(make_rgb_data(w as usize, h as usize));
    let m1 = oklab::rgb_to_lms_matrix(ColorPrimaries::Bt709).unwrap();
    let m1_inv = oklab::lms_to_rgb_matrix(ColorPrimaries::Bt709).unwrap();

    suite.compare(&format!("perpixel_{label}"), |group| {
        group.throughput(Throughput::Elements(n as u64));

        // Pipeline API: scatter → 3 separate filters → gather (strip-processed)
        {
            let src = Arc::clone(&src);
            let mut pipe = Pipeline::new(PipelineConfig::default()).unwrap();
            let mut exposure = zenfilters::filters::Exposure::default();
            exposure.stops = 0.2;
            pipe.push(Box::new(exposure));
            let mut contrast = zenfilters::filters::Contrast::default();
            contrast.amount = 0.15;
            pipe.push(Box::new(contrast));
            let mut sat = zenfilters::filters::Saturation::default();
            sat.factor = 1.1;
            pipe.push(Box::new(sat));

            group.bench("pipeline_3_filters", move |b| {
                let mut ctx = FilterContext::new();
                let mut dst = vec![0.0f32; n * 3];
                b.iter(|| {
                    pipe.apply(&src, &mut dst, w, h, 3, &mut ctx).unwrap();
                });
            });
        }

        // Fused interleaved: RGB→Oklab→adjust→RGB in one streaming SIMD pass
        {
            let src = Arc::clone(&src);
            group.bench("fused_interleaved", move |b| {
                let mut dst = vec![0.0f32; n * 3];
                b.iter(|| {
                    zenfilters::fused_interleaved_adjust(
                        &src,
                        &mut dst,
                        3,
                        &m1,
                        &m1_inv,
                        1.0,
                        1.0,
                        0.01,       // bp
                        1.0 / 0.99, // inv_range
                        1.15,       // wp_exp (exposure + white point)
                        1.08,       // contrast_exp
                        0.97,       // contrast_scale
                        0.15,       // shadows
                        -0.1,       // highlights
                        1.0,        // dehaze_contrast
                        1.0,        // dehaze_chroma
                        1.07,       // exposure_chroma
                        0.002,      // temp_offset
                        -0.001,     // tint_offset
                        1.1,        // sat
                        0.3,        // vib_amount
                        2.0,        // vib_protection
                    );
                });
            });
        }
    });
}

const SIZES: &[(u32, u32, &str)] = &[(1920, 1080, "1080p"), (3840, 2160, "4k")];

zenbench::main!(|suite| {
    for &(w, h, label) in SIZES {
        bench_fused_group(suite, w, h, label);
    }
});
