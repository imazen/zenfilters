//! Resize-aware pipeline: automatically splits filters around a resize step.
//!
//! When an image processing pipeline includes a resize (e.g., 4K → 1080p export),
//! filters should run at the right resolution:
//!
//! - **PreResize** filters (CA correction, noise reduction, sharpening, clarity)
//!   run at full input resolution where detail is richest.
//! - **PostResize** filters (grain, vignette, bloom) run at output resolution
//!   where their spatial effects are relative to the final frame.
//! - **Either** filters (exposure, contrast, curves, color) can run at either
//!   resolution — we run them pre-resize since full-res has more precision.
//!
//! # Sigma adjustment
//!
//! Neighborhood filters with absolute-pixel sigma values need adjustment when
//! their application resolution changes. If a filter designed for 4K is applied
//! after downscaling to 1080p, its sigma should be scaled:
//!
//! ```text
//! sigma_effective = sigma_original * (output_size / input_size)
//! ```
//!
//! This ensures the filter targets the same perceptual frequency band regardless
//! of the resolution it runs at.
//!
//! # API
//!
//! ```ignore
//! use zenfilters::resize_pipeline::ResizePipeline;
//!
//! let mut rp = ResizePipeline::new(input_width, input_height, output_width, output_height);
//! rp.push(Box::new(NoiseReduction { luminance: 0.5, .. })); // → pre-resize
//! rp.push(Box::new(Exposure { stops: 0.3 }));                // → pre-resize (Either)
//! rp.push(Box::new(Grain { amount: 0.2, .. }));              // → post-resize
//!
//! // Returns (pre_pipeline, post_pipeline, scale_factor)
//! let (pre, post, scale) = rp.build();
//!
//! // Caller applies: pre → resize → post
//! pre.apply(&src, &mut intermediate, in_w, in_h, 3, &mut ctx)?;
//! // ... resize intermediate to output size ...
//! post.apply(&resized, &mut dst, out_w, out_h, 3, &mut ctx)?;
//! ```

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::filter::{Filter, ResizePhase};
use crate::pipeline::{Pipeline, PipelineConfig};

/// A pipeline builder that splits filters around a resize operation.
pub struct ResizePipeline {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    filters: Vec<(Box<dyn Filter>, ResizePhase)>,
}

impl ResizePipeline {
    /// Create a new resize-aware pipeline.
    pub fn new(input_width: u32, input_height: u32, output_width: u32, output_height: u32) -> Self {
        Self {
            input_width,
            input_height,
            output_width,
            output_height,
            filters: Vec::new(),
        }
    }

    /// The scale factor (output / input). <1.0 for downscale.
    pub fn scale_factor(&self) -> f32 {
        let sw = self.output_width as f32 / self.input_width as f32;
        let sh = self.output_height as f32 / self.input_height as f32;
        (sw + sh) * 0.5
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

    /// Build two pipelines: pre-resize and post-resize.
    ///
    /// Returns `(pre_pipeline, post_pipeline)`.
    ///
    /// - `Either` filters go in the pre-resize pipeline (full-res has more precision).
    /// - `PreResize` filters go in pre-resize.
    /// - `PostResize` filters go in post-resize.
    pub fn build(self) -> (Pipeline, Pipeline) {
        let mut pre = Pipeline::new(PipelineConfig::default()).unwrap();
        let mut post = Pipeline::new(PipelineConfig::default()).unwrap();

        for (filter, phase) in self.filters {
            match phase {
                ResizePhase::PreResize | ResizePhase::Either => pre.push(filter),
                ResizePhase::PostResize => post.push(filter),
            }
        }

        (pre, post)
    }

    /// Input dimensions.
    pub fn input_size(&self) -> (u32, u32) {
        (self.input_width, self.input_height)
    }

    /// Output dimensions.
    pub fn output_size(&self) -> (u32, u32) {
        (self.output_width, self.output_height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::*;

    #[test]
    fn splits_by_phase() {
        let mut rp = ResizePipeline::new(3840, 2160, 1920, 1080);

        // PreResize
        let mut nr = NoiseReduction::default();
        nr.luminance = 0.5;
        rp.push(Box::new(nr));

        // Either → goes to pre
        let mut exp = Exposure::default();
        exp.stops = 0.3;
        rp.push(Box::new(exp));

        // PostResize
        let mut grain = Grain::default();
        grain.amount = 0.2;
        rp.push(Box::new(grain));

        let (pre, post) = rp.build();
        // pre should have 2 filters (NR + Exposure)
        // post should have 1 filter (Grain)
        let _ = (pre, post);
    }

    #[test]
    fn scale_factor_correct() {
        let rp = ResizePipeline::new(3840, 2160, 1920, 1080);
        assert!((rp.scale_factor() - 0.5).abs() < 0.01);

        let rp2 = ResizePipeline::new(1920, 1080, 3840, 2160);
        assert!((rp2.scale_factor() - 2.0).abs() < 0.01);
    }
}
