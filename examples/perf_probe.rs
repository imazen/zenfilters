//! Performance comparison after box blur optimization.
//! Run: cargo run --release --example perf_probe

use std::sync::Arc;
use zenfilters::filters::*;
use zenfilters::{FilterContext, Pipeline, PipelineConfig};

fn make_linear_rgb(w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut data = Vec::with_capacity(n * 3);
    for y in 0..h {
        for x in 0..w {
            let t = (y * w + x) as f32 / n as f32;
            data.push((t * 0.6 + 0.2).clamp(0.01, 0.99));
            data.push(((1.0 - t) * 0.5 + 0.25).clamp(0.01, 0.99));
            data.push(((x as f32 / w as f32) * 0.4 + 0.3).clamp(0.01, 0.99));
        }
    }
    data
}

fn make_pipeline_clarity() -> Pipeline {
    let mut pipe = Pipeline::new(PipelineConfig::default()).unwrap();
    let mut c = Clarity::default();
    c.sigma = 4.0;
    c.amount = 0.3;
    pipe.push(Box::new(c));
    pipe
}

fn make_pipeline_realistic() -> Pipeline {
    let mut pipe = Pipeline::new(PipelineConfig::default()).unwrap();
    let mut fa = FusedAdjust::default();
    fa.exposure = 0.3;
    fa.contrast = 0.2;
    fa.highlights = 0.4;
    fa.shadows = 0.3;
    fa.saturation = 1.1;
    fa.vibrance = 0.3;
    pipe.push(Box::new(fa));
    let mut cl = Clarity::default();
    cl.sigma = 4.0;
    cl.amount = 0.2;
    pipe.push(Box::new(cl));
    let mut sh = AdaptiveSharpen::default();
    sh.amount = 0.3;
    sh.sigma = 1.2;
    sh.noise_floor = 0.004;
    sh.detail = 0.5;
    sh.masking = 0.3;
    pipe.push(Box::new(sh));
    pipe
}

fn make_pipeline_heavy() -> Pipeline {
    let mut pipe = Pipeline::new(PipelineConfig::default()).unwrap();
    let mut cl = Clarity::default();
    cl.sigma = 4.0;
    cl.amount = 0.3;
    pipe.push(Box::new(cl));
    let mut tx = Texture::default();
    tx.sigma = 1.5;
    tx.amount = 0.3;
    pipe.push(Box::new(tx));
    let mut nr = NoiseReduction::default();
    nr.luminance = 0.5;
    nr.chroma = 0.3;
    nr.scales = 4;
    pipe.push(Box::new(nr));
    pipe
}

fn make_pipeline_perpixel() -> Pipeline {
    let mut pipe = Pipeline::new(PipelineConfig::default()).unwrap();
    let mut fa = FusedAdjust::default();
    fa.exposure = 0.5;
    fa.contrast = 0.3;
    fa.saturation = 1.2;
    fa.vibrance = 0.4;
    pipe.push(Box::new(fa));
    pipe
}

fn bench_pipeline(
    suite: &mut zenbench::Suite,
    name: &str,
    w: u32,
    h: u32,
    make_pipe: fn() -> Pipeline,
) {
    let n = (w as usize) * (h as usize);
    let src: Arc<[f32]> = make_linear_rgb(w as usize, h as usize).into();

    suite.compare(name, |group| {
        let src = Arc::clone(&src);
        group.bench("run", move |b| {
            let src = Arc::clone(&src);
            b.with_input(move || {
                let src = Arc::clone(&src);
                (make_pipe(), src, vec![0.0f32; n * 3], FilterContext::new())
            })
            .run(move |(pipe, src, mut dst, mut ctx)| {
                pipe.apply(&src, &mut dst, w, h, 3, &mut ctx).unwrap();
                (pipe, src, dst, ctx)
            })
        });
    });
}

fn main() {
    zenbench::run(|suite| {
        bench_pipeline(suite, "perpixel_1080p", 1920, 1080, make_pipeline_perpixel);
        bench_pipeline(suite, "clarity_1080p", 1920, 1080, make_pipeline_clarity);
        bench_pipeline(
            suite,
            "realistic_1080p",
            1920,
            1080,
            make_pipeline_realistic,
        );
        bench_pipeline(suite, "heavy_1080p", 1920, 1080, make_pipeline_heavy);

        bench_pipeline(suite, "perpixel_4k", 3840, 2160, make_pipeline_perpixel);
        bench_pipeline(suite, "clarity_4k", 3840, 2160, make_pipeline_clarity);
        bench_pipeline(suite, "realistic_4k", 3840, 2160, make_pipeline_realistic);
        bench_pipeline(suite, "heavy_4k", 3840, 2160, make_pipeline_heavy);
    });
}
