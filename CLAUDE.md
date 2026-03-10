# zenfilters

Oklab perceptual color space image filter library with SIMD dispatch via archmage.

## Goal Set (2026-03-10)

### 1. Feature Parity with Lightroom

Before training a neural model, zenfilters needs all the adjustment capabilities Lightroom offers. Current coverage: 39 filters across exposure, tone, color, detail, and effects.

**Missing filters (high priority for model expressiveness):**
- **Whites/Blacks sliders** — separate from white_point/black_point; LR has both pairs
- **Parametric Tone Curve** — region-based with movable dividers (we have point curve only)
- **Tone Curve Saturation refinement** — per-region saturation on the curve
- **Sharpening Detail + Masking** — LR has 4 sharpening controls (amount, radius, detail, masking); we have amount only
- **Noise Reduction Detail + Contrast** — LR has 5 NR controls (luminance amount/detail/contrast, color amount/detail); we have luminance + chroma amount only
- **B&W Channel Mixer** — per-color grayscale contribution (red, orange, yellow, green, aqua, blue, purple, magenta)
- **Camera Calibration** — RGB primary hue/saturation shifts

**Missing filters (lower priority, harder to implement):**
- **Lens Blur** — AI depth-based bokeh with bokeh shape styles
- **Transform/Upright** — perspective correction (auto, guided, level, vertical, full)
- **Lens Distortion** — barrel/pincushion correction with profiles

### 2. zentract Integration (Neural Model)

Replace or supplement the 64-cluster K-means model with a proper neural network via zentract (ONNX inference).

- **zentract location**: `/home/lilith/work/zen/zentract/`
- **Architecture**: 3-crate workspace (zentract-types, zentract-abi, zentract-api). Uses dlopen to keep tract's 267-crate dep out of host binary.
- **Plan**: Train an MLP (features -> params) in Python, export ONNX, load via zentract at runtime
- **Current cluster model**: 64 clusters, k=3 inverse-distance blend, +3.2 zensim vs baseline
- **Target**: Continuous prediction (no cluster quantization), better generalization

### 3. Better Image Comparison Metric

Build a spatially-aware comparison system that goes beyond global zensim scores.

- **Regional masks**: Divide image into semantically meaningful regions (sky, skin, shadows, highlights, midtones) using masks similar to those in HSL adjust and tone curve
- **Per-region histograms**: Compare L/a/b distributions within each region, not just globally
- **Weighted scoring**: Weight regions by perceptual importance (skin > sky > deep shadows)
- **Use existing mask infrastructure**: Leverage the luminance/chroma masking already in filters like clarity, local_tone_map
- **Goal**: More diagnostic comparison — know WHERE edits diverge, not just that they do

## Known Issues

- No git remote configured yet
