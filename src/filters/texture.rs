use crate::access::ChannelAccess;
use crate::blur::{GaussianKernel, gaussian_blur_plane};
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::planes::OklabPlanes;

/// Texture enhancement: fine detail contrast.
///
/// Similar to Clarity but targets higher-frequency detail (smaller features
/// like skin pores, fabric weave, individual leaves). Uses a finer-scale
/// band extraction than Clarity.
///
/// This mirrors Lightroom's "Texture" slider introduced in 2019.
///
/// ```text
/// fine   = gaussian_blur(L, sigma)
/// coarse = gaussian_blur(L, sigma * 2)
/// detail = fine - coarse      // the texture band
/// L'     = L + amount * detail
/// ```
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Texture {
    /// Sigma for the fine-scale blur. Smaller than Clarity's sigma to
    /// target finer detail. Default: 1.5.
    pub sigma: f32,
    /// Enhancement amount. Positive = sharpen texture, negative = soften.
    /// Typical: 0.3–1.0.
    pub amount: f32,
}

impl Default for Texture {
    fn default() -> Self {
        Self {
            sigma: 1.5,
            amount: 0.0,
        }
    }
}

impl Filter for Texture {
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

        let kernel_fine = GaussianKernel::new(self.sigma);
        let mut blurred_fine = ctx.take_f32(pc);
        gaussian_blur_plane(&planes.l, &mut blurred_fine, w, h, &kernel_fine, ctx);

        let kernel_coarse = GaussianKernel::new(self.sigma * 2.0);
        let mut blurred_coarse = ctx.take_f32(pc);
        gaussian_blur_plane(&planes.l, &mut blurred_coarse, w, h, &kernel_coarse, ctx);

        let amount = self.amount;
        let mut dst = ctx.take_f32(pc);
        for i in 0..pc {
            let detail = blurred_fine[i] - blurred_coarse[i];
            dst[i] = (planes.l[i] + amount * detail).max(0.0);
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

    #[test]
    fn zero_amount_is_identity() {
        let mut planes = OklabPlanes::new(32, 32);
        for (i, v) in planes.l.iter_mut().enumerate() {
            *v = (i as f32 / 1024.0).sin().abs();
        }
        let original = planes.l.clone();
        Texture::default().apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, original);
    }

    #[test]
    fn positive_enhances_detail() {
        let mut planes = OklabPlanes::new(64, 64);
        // Fine checkerboard pattern (high frequency texture)
        for y in 0..64 {
            for x in 0..64 {
                let i = y * 64 + x;
                planes.l[i] = if (x / 2 + y / 2) % 2 == 0 { 0.6 } else { 0.4 };
            }
        }
        let before_std = std_dev(&planes.l);
        let mut tex = Texture::default();
        tex.amount = 0.8;
        tex.apply(&mut planes, &mut FilterContext::new());
        let after_std = std_dev(&planes.l);
        assert!(
            after_std > before_std,
            "texture should increase detail: {before_std} -> {after_std}"
        );
    }

    #[test]
    fn negative_softens() {
        let mut planes = OklabPlanes::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let i = y * 64 + x;
                planes.l[i] = if (x / 2 + y / 2) % 2 == 0 { 0.6 } else { 0.4 };
            }
        }
        let before_std = std_dev(&planes.l);
        let mut tex = Texture::default();
        tex.amount = -0.5;
        tex.apply(&mut planes, &mut FilterContext::new());
        let after_std = std_dev(&planes.l);
        assert!(
            after_std < before_std,
            "negative texture should soften: {before_std} -> {after_std}"
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
        let mut tex = Texture::default();
        tex.amount = 0.5;
        tex.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.a, a_orig);
    }

    fn std_dev(data: &[f32]) -> f32 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance =
            data.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / data.len() as f32;
        variance.sqrt()
    }
}
