# zenfilters

Photo filter pipeline operating in Oklab perceptual color space with SIMD acceleration via [archmage](https://github.com/imazen/archmage).

45+ filters covering Lightroom-equivalent adjustments: exposure, contrast, highlights/shadows, clarity, noise reduction, color grading, tone curves, HSL, vibrance, grain, vignette, and more.

`#![forbid(unsafe_code)]` — entirely safe Rust. SIMD dispatch happens through archmage's compile-time token system.

## How it works

```text
Input (linear RGB f32 or sRGB u8)
  → scatter: deinterleave to planar Oklab (separate L, a, b arrays)
    → filter stack: each filter modifies planes in-place
      → gamut mapping: compress out-of-gamut colors
        → gather: reinterleave to output format
```

Splitting L/a/b into contiguous `Vec<f32>` planes means luminance-only filters (exposure, contrast, tone curves) touch one plane of contiguous floats — ideal for SIMD. Oklab is perceptually uniform, so arithmetic operations produce visually proportional changes without the nonlinear surprises of sRGB or even Lab.

Processing happens in L3-cache-friendly horizontal strips. Each strip is scattered, filtered, and gathered before moving on, keeping working data under ~4 MB. Neighborhood filters (clarity, sharpen, denoise) use overlapping strips with halo rows — no full-frame materialization needed.

## Usage

```rust
use zenfilters::{Pipeline, PipelineConfig, FilterContext};
use zenfilters::filters::*;

let mut pipeline = Pipeline::new(PipelineConfig::default()).unwrap();

pipeline.push(Box::new(Exposure { stops: 0.5 }));
pipeline.push(Box::new(Clarity { sigma: 4.0, amount: 0.3 }));
pipeline.push(Box::new(Vibrance { amount: 0.4, protection: 0.8 }));

// Reusable context eliminates per-call allocations
let mut ctx = FilterContext::new();

let (w, h) = (1920, 1080);
let src = vec![0.5f32; w * h * 3]; // interleaved linear RGB f32
let mut dst = vec![0.0f32; w * h * 3];
pipeline.apply(&src, &mut dst, w as u32, h as u32, 3, &mut ctx).unwrap();
```

The `buffer` feature adds a convenience API that handles format conversion (sRGB u8, HDR PQ, Display P3) automatically via `PipelineBufferExt`.

## Filters

### Tone & Exposure
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `Exposure` | `stops` | Linear light exposure in stops |
| `AutoExposure` | `strength`, `target`, `max_correction` | Geometric mean normalization |
| `Contrast` | `amount` | Midtone-pivoted contrast |
| `HighlightsShadows` | `highlights`, `shadows` | Highlight/shadow recovery |
| `WhitesBlacks` | `whites`, `blacks` | Extreme luminance control (smoothstep-weighted) |
| `BlackPoint` / `WhitePoint` | `level` | Black/white level remap |
| `HighlightRecovery` | `strength` | Dedicated clipping recovery |
| `ShadowLift` | `strength` | Dedicated shadow recovery |
| `ToneCurve` | control points or LUT | Monotone cubic Hermite interpolation |
| `ParametricCurve` | 4 zones, 3 dividers | Lightroom-style parametric curve |
| `ChannelCurves` | per-channel R/G/B LUTs | sRGB-space per-channel curves |
| `Sigmoid` | `contrast`, `skew`, `chroma_compression` | Generalized sigmoid tone mapper |
| `BasecurveToneMap` | preset-based | Camera-specific tone curves |
| `DtSigmoid` | `contrast`, `skew` | darktable-compatible sigmoid |
| `LocalToneMap` | `compression`, `detail_boost`, `sigma` | Base/detail decomposition (neighborhood) |

### Sharpening & Detail
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `AdaptiveSharpen` | `amount`, `sigma`, `noise_floor`, `detail`, `masking` | Noise-gated unsharp mask with edge masking |
| `Sharpen` | `sigma`, `amount` | Basic unsharp mask |
| `Clarity` | `sigma`, `amount` | Two-band mid-frequency local contrast |
| `Texture` | `sigma`, `amount` | Fine detail enhancement |

### Noise Reduction
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `NoiseReduction` | `luminance`, `chroma`, `detail`, `luminance_contrast`, `chroma_detail` | Wavelet (a trous) with soft thresholding |
| `Bilateral` | `spatial_sigma`, `range_sigma`, `strength` | Edge-preserving bilateral filter |
| `Blur` | `sigma` | Gaussian blur |

### Color
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `Temperature` | `shift` | Color temperature via b-channel offset |
| `Tint` | `shift` | Green-magenta tint via a-channel offset |
| `Saturation` | `factor` | Linear saturation scaling |
| `Vibrance` | `amount`, `protection` | Smart saturation (protects already-saturated colors) |
| `HueRotate` | `shift` | Hue rotation in a/b plane |
| `HslAdjust` | per-range hue/sat/lum (8 ranges) | Selective per-hue-range adjustments |
| `ColorGrading` | shadow/midtone/highlight tints | Split-tone color grading |
| `CameraCalibration` | R/G/B hue+sat, shadow tint | Camera primary calibration |
| `ColorMatrix` | 5x5 matrix | Arbitrary color matrix in linear RGB |
| `Cat16` | source/target primaries | Chromatic adaptation (CAT16) |
| `GamutExpand` | `strength`, `knee` | Soft chroma expansion toward gamut boundary |
| `BwMixer` | 8 per-color luminance weights | Chroma-aware B&W channel mixer |

### Effects
| Filter | Parameters | Description |
|--------|-----------|-------------|
| `Grain` | `amount`, `size`, `seed` | Deterministic film grain |
| `Vignette` | `strength`, `exponent` | Radial edge darkening |
| `Devignette` | `strength`, `exponent` | Lens vignette correction |
| `Dehaze` | `strength` | Contrast and chroma boost |
| `ChromaticAberration` | `shift_a`, `shift_b` | Lateral CA correction |
| `Grayscale` / `Sepia` / `Invert` | — | Standard effects |

### Performance
| Filter | Description |
|--------|-------------|
| `FusedAdjust` | Combines 11 per-pixel operations (exposure, contrast, H/S, dehaze, temperature, tint, saturation, vibrance, BP/WP) in a single SIMD pass |

## Strip processing

All processing (including pipelines with neighborhood filters) uses horizontal strip processing to stay L3-cache-friendly.

For per-pixel-only pipelines, strips are independent — scatter, filter, gather, move on. For pipelines with neighborhood filters, each strip is extended by a halo of extra rows on each side. The halo size is the sum of all neighborhood radii in the pipeline. Each filter "consumes" its radius of correct context from the previous filter's output, so the core rows are always correct.

At 4K width with a typical clarity + sharpen pipeline (halo ~50px, strip ~130 core rows), the working set is ~9 MB per strip vs ~100 MB for the full frame.

## Gamut mapping

Three strategies for handling out-of-gamut colors after filtering:

- **Clip** (default) — clamp negative RGB to 0. Fast; sufficient for most adjustments.
- **ChromaReduce** — bisection to reduce chroma until in-gamut. Preserves hue.
- **SoftCompress** — precomputed LUT of max chroma per (L, hue), with a smooth rational knee function. Best quality for aggressive saturation boosts.

## Color space support

The pipeline supports BT.709 (sRGB), Display P3, and BT.2020 primaries. HDR content is handled via reference white normalization — set `reference_white` to 203.0 for PQ (ITU-R BT.2408) so that filters operate in a normalized [0, 1] range regardless of absolute luminance.

## Auto-tuning (experimental)

The `experimental` feature enables automatic photo enhancement:

1. Extract 142-dim histogram features from an image (luminance zones, chroma zones, hue sectors)
2. K-means lookup against 64 pre-trained clusters (in `data/`)
3. Inverse-distance blend of k=3 nearest cluster parameters into 18 filter params

Trained on the MIT-Adobe FiveK dataset using Nelder-Mead optimization with zensim (perceptual similarity) as the loss function. Also includes a rule-based heuristic fallback.

## License

AGPL-3.0-or-later
