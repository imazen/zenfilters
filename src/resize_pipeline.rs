//! Resize-aware pipeline: splits filters around crop, resize, and canvas steps.
//!
//! When processing includes geometry changes (crop, resize, orient, pad), filters
//! must run at the correct stage:
//!
//! ```text
//! Decode → [PreResize filters] → Crop → Resize → Orient → [PostResize filters] → Pad/Canvas → Encode
//! ```
//!
//! - **PreResize** filters run on the full source (or cropped source) before downscale.
//!   These need full-resolution detail: CA correction, noise reduction, sharpening, clarity.
//! - **PostResize** filters run on the final output canvas. These are spatial effects
//!   relative to the viewer: grain, vignette, bloom.
//! - **Either** filters (per-pixel: exposure, contrast, curves) go pre-resize by default
//!   for maximum precision, but can be forced post-resize for speed.
//!
//! # Integration with zenlayout
//!
//! This module accepts a [`LayoutSpec`] that describes the geometry transformation.
//! zenlayout's `LayoutPlan` can be converted to `LayoutSpec`:
//!
//! ```ignore
//! use zenfilters::resize_pipeline::{ResizePipeline, LayoutSpec};
//!
//! let spec = LayoutSpec {
//!     source_width: 4000,
//!     source_height: 3000,
//!     crop: Some(CropRect { x: 100, y: 100, w: 3800, h: 2800 }),
//!     output_width: 1920,
//!     output_height: 1080,
//!     canvas_width: 1920,  // may differ from output if padded
//!     canvas_height: 1080,
//!     placement_x: 0,      // offset on canvas (non-zero when padded)
//!     placement_y: 0,
//! };
//!
//! let mut rp = ResizePipeline::from_layout(spec);
//! rp.push(Box::new(NoiseReduction { .. }));  // → pre-resize
//! rp.push(Box::new(Grain { .. }));           // → post-resize
//!
//! let plan = rp.build();
//! // plan.pre_filters  → apply at source_width × source_height (or cropped)
//! // plan.post_filters → apply at canvas_width × canvas_height
//! // plan.scale_factor → for sigma adjustment
//! ```
//!
//! # Without zenlayout
//!
//! For simple resize-only workflows:
//! ```ignore
//! let rp = ResizePipeline::simple(3840, 2160, 1920, 1080);
//! ```

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::filter::{Filter, ResizePhase};
use crate::pipeline::{Pipeline, PipelineConfig};

/// Geometry specification describing crop, resize, and canvas placement.
///
/// This is the information zenfilters needs from a layout engine (zenlayout)
/// to correctly split filters into pre-resize and post-resize groups.
///
/// zenlayout users: convert `LayoutPlan` to `LayoutSpec` via the provided
/// `From` impl in the zenlayout integration, or construct manually.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LayoutSpec {
    /// Source image dimensions (before any processing).
    pub source_width: u32,
    pub source_height: u32,

    /// Optional crop region on the source (applied before resize).
    pub crop: Option<CropRect>,

    /// Resize target dimensions (after crop, before canvas placement).
    pub output_width: u32,
    pub output_height: u32,

    /// Final canvas dimensions (may be larger than output if padded).
    pub canvas_width: u32,
    pub canvas_height: u32,

    /// Placement offset on canvas (non-zero when image is centered on a padded canvas).
    pub placement_x: u32,
    pub placement_y: u32,
}

/// A crop rectangle in source pixel coordinates.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl LayoutSpec {
    /// Scale factor (output / source). Accounts for crop.
    pub fn scale_factor(&self) -> f32 {
        let src_w = self.crop.as_ref().map_or(self.source_width, |c| c.w);
        let src_h = self.crop.as_ref().map_or(self.source_height, |c| c.h);
        let sw = self.output_width as f32 / src_w as f32;
        let sh = self.output_height as f32 / src_h as f32;
        (sw + sh) * 0.5
    }

    /// Whether this layout includes any resize (not just crop/pad).
    pub fn has_resize(&self) -> bool {
        let src_w = self.crop.as_ref().map_or(self.source_width, |c| c.w);
        let src_h = self.crop.as_ref().map_or(self.source_height, |c| c.h);
        src_w != self.output_width || src_h != self.output_height
    }

    /// Whether this layout includes canvas padding.
    pub fn has_padding(&self) -> bool {
        self.canvas_width != self.output_width || self.canvas_height != self.output_height
    }

    /// Dimensions that pre-resize filters operate on (source or crop).
    pub fn pre_resize_dims(&self) -> (u32, u32) {
        match &self.crop {
            Some(c) => (c.w, c.h),
            None => (self.source_width, self.source_height),
        }
    }

    /// Dimensions that post-resize filters operate on (canvas).
    pub fn post_resize_dims(&self) -> (u32, u32) {
        (self.canvas_width, self.canvas_height)
    }
}

/// Built plan with split pipelines and layout information.
pub struct ResizePlan {
    /// Filters to apply at source resolution (before resize).
    pub pre: Pipeline,
    /// Filters to apply at output/canvas resolution (after resize).
    pub post: Pipeline,
    /// The layout specification.
    pub layout: LayoutSpec,
}

/// A pipeline builder that splits filters around geometry operations.
pub struct ResizePipeline {
    layout: LayoutSpec,
    filters: Vec<(Box<dyn Filter>, ResizePhase)>,
}

impl ResizePipeline {
    /// Create from a full layout specification.
    pub fn from_layout(layout: LayoutSpec) -> Self {
        Self {
            layout,
            filters: Vec::new(),
        }
    }

    /// Create for a simple resize (no crop, no padding).
    pub fn simple(
        source_width: u32,
        source_height: u32,
        output_width: u32,
        output_height: u32,
    ) -> Self {
        Self::from_layout(LayoutSpec {
            source_width,
            source_height,
            crop: None,
            output_width,
            output_height,
            canvas_width: output_width,
            canvas_height: output_height,
            placement_x: 0,
            placement_y: 0,
        })
    }

    /// The scale factor (output / source).
    pub fn scale_factor(&self) -> f32 {
        self.layout.scale_factor()
    }

    /// Add a filter. Its `resize_phase()` determines placement.
    pub fn push(&mut self, filter: Box<dyn Filter>) {
        let phase = filter.resize_phase();
        self.filters.push((filter, phase));
    }

    /// Add a filter with an explicit phase override.
    pub fn push_at(&mut self, filter: Box<dyn Filter>, phase: ResizePhase) {
        self.filters.push((filter, phase));
    }

    /// Build the split pipelines.
    ///
    /// Returns a [`ResizePlan`] with pre-resize and post-resize pipelines
    /// plus the layout spec for the caller to execute the resize step.
    pub fn build(self) -> ResizePlan {
        let mut pre = Pipeline::new(PipelineConfig::default()).unwrap();
        let mut post = Pipeline::new(PipelineConfig::default()).unwrap();

        for (filter, phase) in self.filters {
            match phase {
                ResizePhase::PreResize | ResizePhase::Either => pre.push(filter),
                ResizePhase::PostResize => post.push(filter),
            }
        }

        ResizePlan {
            pre,
            post,
            layout: self.layout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::*;

    #[test]
    fn simple_splits_by_phase() {
        let mut rp = ResizePipeline::simple(3840, 2160, 1920, 1080);

        let mut nr = NoiseReduction::default();
        nr.luminance = 0.5;
        rp.push(Box::new(nr));

        let mut exp = Exposure::default();
        exp.stops = 0.3;
        rp.push(Box::new(exp));

        let mut grain = Grain::default();
        grain.amount = 0.2;
        rp.push(Box::new(grain));

        let plan = rp.build();
        assert!((plan.layout.scale_factor() - 0.5).abs() < 0.01);
        let _ = (plan.pre, plan.post);
    }

    #[test]
    fn layout_with_crop() {
        let spec = LayoutSpec {
            source_width: 4000,
            source_height: 3000,
            crop: Some(CropRect {
                x: 500,
                y: 375,
                w: 3000,
                h: 2250,
            }),
            output_width: 1500,
            output_height: 1125,
            canvas_width: 1500,
            canvas_height: 1125,
            placement_x: 0,
            placement_y: 0,
        };

        assert!(spec.has_resize());
        assert!(!spec.has_padding());
        assert!((spec.scale_factor() - 0.5).abs() < 0.01);
        assert_eq!(spec.pre_resize_dims(), (3000, 2250));
        assert_eq!(spec.post_resize_dims(), (1500, 1125));
    }

    #[test]
    fn layout_with_padding() {
        let spec = LayoutSpec {
            source_width: 1920,
            source_height: 1080,
            crop: None,
            output_width: 800,
            output_height: 600,
            canvas_width: 1000,
            canvas_height: 600,
            placement_x: 100,
            placement_y: 0,
        };

        assert!(spec.has_resize());
        assert!(spec.has_padding());
        assert_eq!(spec.post_resize_dims(), (1000, 600));
    }
}
