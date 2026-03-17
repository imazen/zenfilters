#![forbid(unsafe_code)]
//! Photo filter operations on planar Oklab f32 data with SIMD acceleration.
//!
//! zenfilters provides a pipeline for applying photo adjustments (exposure,
//! contrast, clarity, saturation, etc.) in Oklab color space using a planar
//! layout for maximum SIMD throughput.
//!
//! ## Pipeline
//!
//! ```text
//! Linear RGB f32 → scatter to planar Oklab → filters → gather to Linear RGB f32
//! ```
//!
//! Filters operate on separate L, a, b planes. Per-pixel L-only filters
//! (exposure, contrast) run at full SIMD width on contiguous memory.
//! Neighborhood filters (clarity, brilliance) use separable Gaussian blur
//! on the L plane for 188× speedup over naive interleaved approaches.
//!
//! ## Usage
//!
//! ```
//! use zenfilters::{Pipeline, PipelineConfig, FilterContext, OklabPlanes};
//! use zenfilters::filters::*;
//! use zenpixels::ColorPrimaries;
//!
//! let mut pipeline = Pipeline::new(PipelineConfig::default()).unwrap();
//!
//! let mut exposure = Exposure::default();
//! exposure.stops = 0.5;
//! pipeline.push(Box::new(exposure));
//!
//! let mut clarity = Clarity::default();
//! clarity.amount = 0.3;
//! pipeline.push(Box::new(clarity));
//!
//! let mut vibrance = Vibrance::default();
//! vibrance.amount = 0.4;
//! pipeline.push(Box::new(vibrance));
//!
//! // Create a reusable context to avoid per-call allocations
//! let mut ctx = FilterContext::new();
//!
//! // Apply to interleaved linear RGB f32 data
//! let (w, h) = (64, 64);
//! let src = vec![0.5f32; w * h * 3];
//! let mut dst = vec![0.0f32; w * h * 3];
//! pipeline.apply(&src, &mut dst, w as u32, h as u32, 3, &mut ctx).unwrap();
//! ```

extern crate alloc;

whereat::define_at_crate_info!();

mod access;
mod blur;
mod context;
mod filter;
pub mod filters;
mod gamut_lut;
mod gamut_map;
mod pipeline;
mod planes;
pub mod regional;
mod scatter_gather;
mod simd;

#[cfg(feature = "buffer")]
mod convenience;
#[cfg(feature = "srgb-filters")]
#[allow(clippy::manual_clamp)]
pub mod srgb_filters;

pub use access::ChannelAccess;
pub use blur::GaussianKernel;

/// Internal blur functions exposed for benchmarking. Not part of the public API.
#[cfg(feature = "experimental")]
#[doc(hidden)]
pub mod blur_internals {
    pub use crate::blur::{
        DericheCoefficients, ExtendedBoxBlur, GaussianKernel, deriche_blur_plane,
        extended_box_blur_plane, gaussian_blur_plane, gaussian_blur_plane_scalar, kernel_sigma,
        sigma_to_stackblur_radius, stackblur_plane,
    };
}
pub use context::FilterContext;
#[cfg(feature = "buffer")]
pub use convenience::{ConvenienceError, PipelineBufferExt, apply_to_buffer};
pub use filter::Filter;
pub use gamut_map::GamutMapping;
pub use pipeline::{Pipeline, PipelineConfig, PipelineError};
pub use planes::OklabPlanes;
pub use scatter_gather::{
    gather_from_oklab, gather_oklab_to_srgb_u8, scatter_srgb_u8_to_oklab, scatter_to_oklab,
};

/// Fused interleaved per-pixel adjust: RGB→Oklab→adjust→RGB in one SIMD pass.
#[cfg(feature = "experimental")]
#[allow(clippy::too_many_arguments)]
pub fn fused_interleaved_adjust(
    src: &[f32],
    dst: &mut [f32],
    channels: u32,
    m1: &zenpixels_convert::gamut::GamutMatrix,
    m1_inv: &zenpixels_convert::gamut::GamutMatrix,
    inv_white: f32,
    reference_white: f32,
    bp: f32,
    inv_range: f32,
    wp_exp: f32,
    contrast_exp: f32,
    contrast_scale: f32,
    shadows: f32,
    highlights: f32,
    dehaze_contrast: f32,
    dehaze_chroma: f32,
    exposure_chroma: f32,
    temp_offset: f32,
    tint_offset: f32,
    sat: f32,
    vib_amount: f32,
    vib_protection: f32,
) {
    simd::fused_interleaved_adjust(
        src,
        dst,
        channels,
        m1,
        m1_inv,
        inv_white,
        reference_white,
        bp,
        inv_range,
        wp_exp,
        contrast_exp,
        contrast_scale,
        shadows,
        highlights,
        dehaze_contrast,
        dehaze_chroma,
        exposure_chroma,
        temp_offset,
        tint_offset,
        sat,
        vib_amount,
        vib_protection,
    );
}
