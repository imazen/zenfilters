use crate::access::ChannelAccess;
use crate::blur::{GaussianKernel, gaussian_blur_plane};
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::planes::OklabPlanes;

/// Clarity: multi-scale local contrast enhancement on L channel.
///
/// Uses a two-band decomposition to isolate the mid-frequency "clarity"
/// band, avoiding both noise amplification (from fine detail) and halos
/// (from coarse edges):
///
/// ```text
/// fine   = gaussian_blur(L, sigma)
/// coarse = gaussian_blur(L, sigma * 4)
/// mid    = fine - coarse          // the clarity band
/// L'     = L + amount * mid
/// ```
///
/// This is significantly better than single-scale unsharp mask because
/// it only boosts mid-frequency texture (skin pores, fabric, foliage),
/// not high-frequency noise or low-frequency tonal gradients.
///
/// Inspired by darktable's local contrast module (local Laplacian filter).
/// This multi-scale approach is simpler but captures most of the benefit.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Clarity {
    /// Sigma for the fine-scale blur. Controls the smallest features
    /// affected. Typical: 2.0-8.0. The coarse blur is 4× this.
    pub sigma: f32,
    /// Enhancement amount. Positive = enhance texture, negative = soften.
    /// Typical: 0.3-1.0 for natural results.
    pub amount: f32,
}

impl Default for Clarity {
    fn default() -> Self {
        Self {
            sigma: 4.0,
            amount: 0.0,
        }
    }
}

impl Filter for Clarity {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::L_ONLY
    }

    fn is_neighborhood(&self) -> bool {
        true
    }

    fn apply(&self, planes: &mut OklabPlanes, ctx: &mut FilterContext) {
        if self.amount.abs() < 1e-6 {
            return;
        }

        let pc = planes.pixel_count();
        let w = planes.width;
        let h = planes.height;

        // Fine blur: captures detail above sigma
        let kernel_fine = GaussianKernel::new(self.sigma);
        let mut blurred_fine = ctx.take_f32(pc);
        gaussian_blur_plane(&planes.l, &mut blurred_fine, w, h, &kernel_fine, ctx);

        // Coarse blur: captures structure above sigma*4
        let kernel_coarse = GaussianKernel::new(self.sigma * 4.0);
        let mut blurred_coarse = ctx.take_f32(pc);
        gaussian_blur_plane(&planes.l, &mut blurred_coarse, w, h, &kernel_coarse, ctx);

        // Mid band = fine - coarse; apply: L' = L + amount * mid
        // L' = L + amount * (fine - coarse)
        // Rewrite as unsharp between blurred_fine and blurred_coarse:
        //   dst = blurred_fine + amount * (blurred_fine - blurred_coarse)
        //   Then final = L - blurred_fine + dst = L + amount * (fine - coarse)
        //
        // Simpler: just compute it directly with scale+offset.
        let amount = self.amount;
        let mut dst = ctx.take_f32(pc);
        for i in 0..pc {
            let mid = blurred_fine[i] - blurred_coarse[i];
            dst[i] = (planes.l[i] + amount * mid).max(0.0);
        }

        ctx.return_f32(blurred_fine);
        ctx.return_f32(blurred_coarse);
        let old_l = core::mem::replace(&mut planes.l, dst);
        ctx.return_f32(old_l);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::FilterContext;

    #[test]
    fn zero_amount_is_identity() {
        let mut planes = OklabPlanes::new(32, 32);
        for (i, v) in planes.l.iter_mut().enumerate() {
            *v = (i as f32 / 1024.0).sin().abs();
        }
        let original = planes.l.clone();
        Clarity {
            sigma: 4.0,
            amount: 0.0,
        }
        .apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, original);
    }

    #[test]
    fn positive_amount_enhances_contrast() {
        let mut planes = OklabPlanes::new(64, 64);
        // Create a pattern with local variation at mid-frequency
        for y in 0..64 {
            for x in 0..64 {
                let i = y * 64 + x;
                planes.l[i] = if (x / 8 + y / 8) % 2 == 0 { 0.7 } else { 0.3 };
            }
        }
        let before_std = std_dev(&planes.l);
        Clarity {
            sigma: 3.0,
            amount: 0.5,
        }
        .apply(&mut planes, &mut FilterContext::new());
        let after_std = std_dev(&planes.l);
        assert!(
            after_std > before_std,
            "clarity should increase local contrast: {before_std} -> {after_std}"
        );
    }

    #[test]
    fn does_not_modify_chroma() {
        let mut planes = OklabPlanes::new(32, 32);
        for v in &mut planes.l {
            *v = 0.5;
        }
        for v in &mut planes.a {
            *v = 0.1;
        }
        let a_orig = planes.a.clone();
        Clarity {
            sigma: 3.0,
            amount: 0.5,
        }
        .apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.a, a_orig);
    }

    #[test]
    fn uniform_image_unchanged() {
        // A perfectly uniform image has no mid-frequency content — clarity
        // should produce zero change regardless of amount.
        let mut planes = OklabPlanes::new(32, 32);
        for v in &mut planes.l {
            *v = 0.6;
        }
        let original = planes.l.clone();
        Clarity {
            sigma: 4.0,
            amount: 1.0,
        }
        .apply(&mut planes, &mut FilterContext::new());
        for (a, b) in planes.l.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "uniform image should be unchanged: {a} vs {b}"
            );
        }
    }

    #[test]
    fn negative_amount_softens() {
        let mut planes = OklabPlanes::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let i = y * 64 + x;
                planes.l[i] = if (x / 8 + y / 8) % 2 == 0 { 0.7 } else { 0.3 };
            }
        }
        let before_std = std_dev(&planes.l);
        Clarity {
            sigma: 3.0,
            amount: -0.5,
        }
        .apply(&mut planes, &mut FilterContext::new());
        let after_std = std_dev(&planes.l);
        assert!(
            after_std < before_std,
            "negative amount should soften: {before_std} -> {after_std}"
        );
    }

    fn std_dev(data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance =
            data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }
}
