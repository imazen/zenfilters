# zenfilters

Oklab perceptual color space image filter library with SIMD dispatch via archmage.

## Goal Set (2026-03-10)

### 1. Feature Parity with Lightroom

Before training a neural model, zenfilters needs all the adjustment capabilities Lightroom offers. Current coverage: 45 filters across exposure, tone, color, detail, and effects.

**DONE (high priority, completed 2026-03-10):**
- ~~Whites/Blacks sliders~~ → `WhitesBlacks` (smoothstep-weighted extreme luminance control)
- ~~Parametric Tone Curve~~ → `ParametricCurve` (4 zones, 3 movable dividers, LUT-based)
- ~~Sharpening Detail + Masking~~ → `AdaptiveSharpen` now has `detail` + `masking` fields (4 controls)
- ~~Noise Reduction Detail + Contrast~~ → `NoiseReduction` now has `luminance_contrast` + `chroma_detail` (5 controls)
- ~~B&W Channel Mixer~~ → `BwMixer` (8 per-color luminance weights, chroma-aware)
- ~~Camera Calibration~~ → `CameraCalibration` (R/G/B primary hue+sat shifts, shadow tint)

**DONE (2026-03-18, GEGL gap analysis):**
- ~~Median Blur~~ → `MedianBlur` (neighborhood median, L-only or all channels, radius 1-5)
- ~~Edge Detection~~ → `EdgeDetect` (Sobel + Laplacian, gradient magnitude on L, configurable strength)
- ~~Geometric Transform~~ → `Warp` (experimental, 3×3 projective matrix, bilinear interp, rotation/deskew/affine/perspective)
- ~~Masked Filter~~ → `masked::MaskedFilter` (linear gradient, radial gradient, luminance range masks)

**Still missing (lower priority or needs external data):**
- **Tone Curve Saturation refinement** — per-region saturation on the curve
- **Lens Blur** — AI depth-based bokeh with bokeh shape styles
- **Transform/Upright** — perspective correction (auto, guided, level, vertical, full). Warp provides raw matrix support; needs auto-detection via edge analysis.
- **Lens Distortion** — barrel/pincushion correction with profiles
- **Blend Layers** — Oklab-space compositing of two planes with blend modes (design notes in `masked.rs`)

### 2. zentract Integration (Neural Model)

Replace or supplement the 64-cluster K-means model with a proper neural network via zentract (ONNX inference).

- **zentract location**: `/home/lilith/work/zen/zentract/`
- **Architecture**: 3-crate workspace (zentract-types, zentract-abi, zentract-api). Uses dlopen to keep tract's 267-crate dep out of host binary.
- **Plan**: Train an MLP (features -> params) in Python, export ONNX, load via zentract at runtime
- **Current cluster model**: 64 clusters, k=3 inverse-distance blend, +3.2 zensim vs baseline
- **Target**: Continuous prediction (no cluster quantization), better generalization

### 3. Better Image Comparison Metric

**DONE (core infrastructure, 2026-03-10):**
- `regional.rs` module: `RegionalFeatures::extract()` + `RegionalComparison::compare()`
- 5 luminance zones × 32-bin L histograms + chroma mean
- 4 chroma zones × 32-bin L histograms
- 6 hue sectors × 32-bin a + b histograms
- Weighted aggregate score (midtones > extremes, skin > sky, saturated > neutral)

**TODO:** Integrate into parity/comparison examples, validate against zensim on real data

## Known Issues

- No git remote configured yet
