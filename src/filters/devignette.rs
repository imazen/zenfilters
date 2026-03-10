use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::planes::OklabPlanes;

/// Lens vignetting correction (devignette).
///
/// Compensates for the natural light falloff at the edges of a lens.
/// Applies a radial brightness correction that increases toward the corners,
/// based on the cos^4 law of illumination falloff.
///
/// This is the inverse of the vignette filter — it brightens edges rather
/// than darkening them.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Devignette {
    /// Correction strength. 1.0 = full cos^4 compensation.
    /// Typical: 0.5–1.0. Default: 0.0 (disabled).
    pub strength: f32,
    /// Falloff exponent. Higher values concentrate correction toward corners.
    /// Default: 4.0 (cos^4 law).
    pub exponent: f32,
}

impl Default for Devignette {
    fn default() -> Self {
        Self {
            strength: 0.0,
            exponent: 4.0,
        }
    }
}

impl Filter for Devignette {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::L_AND_CHROMA
    }

    fn apply(&self, planes: &mut OklabPlanes, _ctx: &mut FilterContext) {
        if self.strength.abs() < 1e-6 {
            return;
        }

        let w = planes.width as usize;
        let h = planes.height as usize;
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let diag = (cx * cx + cy * cy).sqrt();
        if diag < 1.0 {
            return;
        }
        let inv_diag = 1.0 / diag;
        let exp = self.exponent;

        for y in 0..h {
            let dy = (y as f32 + 0.5 - cy) * inv_diag;
            for x in 0..w {
                let dx = (x as f32 + 0.5 - cx) * inv_diag;
                let r2 = dx * dx + dy * dy;
                // cos^n falloff approximation: correction = 1 / (1 - strength * r^exp)
                // Simplified: factor = 1 + strength * r^(exp/2)
                let r_pow = r2.powf(exp * 0.5);
                let factor = 1.0 + self.strength * r_pow;

                let idx = y * w + x;
                planes.l[idx] *= factor;
                planes.a[idx] *= factor;
                planes.b[idx] *= factor;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_is_identity() {
        let mut planes = OklabPlanes::new(32, 32);
        for v in &mut planes.l {
            *v = 0.5;
        }
        let orig = planes.l.clone();
        Devignette::default().apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, orig);
    }

    #[test]
    fn brightens_edges() {
        let mut planes = OklabPlanes::new(64, 64);
        for v in &mut planes.l {
            *v = 0.5;
        }
        let mut dv = Devignette::default();
        dv.strength = 0.5;
        dv.apply(&mut planes, &mut FilterContext::new());

        let center = planes.l[32 * 64 + 32];
        let corner = planes.l[0];
        assert!(
            corner > center,
            "corner ({corner}) should be brighter than center ({center})"
        );
    }

    #[test]
    fn center_minimally_affected() {
        let mut planes = OklabPlanes::new(64, 64);
        for v in &mut planes.l {
            *v = 0.5;
        }
        let mut dv = Devignette::default();
        dv.strength = 0.5;
        dv.apply(&mut planes, &mut FilterContext::new());

        let center = planes.l[32 * 64 + 32];
        assert!(
            (center - 0.5).abs() < 0.02,
            "center should be near original: {center}"
        );
    }

    #[test]
    fn scales_chroma() {
        let mut planes = OklabPlanes::new(64, 64);
        for v in &mut planes.l {
            *v = 0.5;
        }
        for v in &mut planes.a {
            *v = 0.1;
        }
        let mut dv = Devignette::default();
        dv.strength = 0.5;
        dv.apply(&mut planes, &mut FilterContext::new());

        let corner_a = planes.a[0];
        let center_a = planes.a[32 * 64 + 32];
        assert!(
            corner_a > center_a,
            "corner chroma should be boosted: {corner_a} vs {center_a}"
        );
    }
}
