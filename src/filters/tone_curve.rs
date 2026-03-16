use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::planes::OklabPlanes;

/// Arbitrary tone curve via control points with cubic spline interpolation.
///
/// Control points define an input→output mapping on the L channel.
/// Between points, monotone cubic Hermite interpolation (Fritsch-Carlson)
/// ensures smooth, monotonic transitions without overshooting.
///
/// This is the equivalent of Lightroom's Tone Curve panel or darktable's
/// tone curve module.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ToneCurve {
    /// LUT with 256 entries mapping input L [0,1] to output L [0,1].
    /// Built from control points at construction time.
    lut: [f32; 256],
}

impl Default for ToneCurve {
    fn default() -> Self {
        let mut lut = [0.0f32; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        Self { lut }
    }
}

impl ToneCurve {
    /// Create a tone curve from control points.
    ///
    /// Points are `(input, output)` pairs in [0, 1]. They must be sorted
    /// by input value. If fewer than 2 points are provided, returns identity.
    ///
    /// Endpoints (0,0) and (1,1) are added automatically if not present.
    pub fn from_points(points: &[(f32, f32)]) -> Self {
        if points.len() < 2 {
            return Self::default();
        }

        let mut pts: Vec<(f32, f32)> = Vec::new();

        // Ensure we have a point at 0
        if points[0].0 > 0.001 {
            pts.push((0.0, 0.0));
        }
        pts.extend_from_slice(points);
        // Ensure we have a point at 1
        if pts.last().unwrap().0 < 0.999 {
            pts.push((1.0, 1.0));
        }

        let lut = build_monotone_cubic_lut(&pts);
        Self { lut }
    }

    /// Create a tone curve from a pre-computed LUT.
    ///
    /// Accepts any number of entries (resampled to 256 via linear interpolation).
    /// Input values are clamped to \[0, 1\].
    pub fn from_lut(lut_data: &[f32]) -> Self {
        if lut_data.len() < 2 {
            return Self::default();
        }

        let mut lut = [0.0f32; 256];
        let src_max = (lut_data.len() - 1) as f32;

        for (i, v) in lut.iter_mut().enumerate() {
            let t = i as f32 / 255.0; // position in [0, 1]
            let src_idx = t * src_max;
            let lo = src_idx as usize;
            let hi = (lo + 1).min(lut_data.len() - 1);
            let frac = src_idx - lo as f32;
            *v = (lut_data[lo] * (1.0 - frac) + lut_data[hi] * frac).clamp(0.0, 1.0);
        }

        Self { lut }
    }

    fn is_identity(&self) -> bool {
        self.lut
            .iter()
            .enumerate()
            .all(|(i, &v)| (v - i as f32 / 255.0).abs() < 1e-4)
    }
}

/// Build a 256-entry LUT from control points using monotone cubic Hermite
/// interpolation (Fritsch-Carlson method).
fn build_monotone_cubic_lut(pts: &[(f32, f32)]) -> [f32; 256] {
    let n = pts.len();
    let mut lut = [0.0f32; 256];

    if n < 2 {
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as f32 / 255.0;
        }
        return lut;
    }

    // Compute secants
    let mut delta = vec![0.0f32; n - 1];
    for i in 0..n - 1 {
        let dx = pts[i + 1].0 - pts[i].0;
        delta[i] = if dx.abs() > 1e-10 {
            (pts[i + 1].1 - pts[i].1) / dx
        } else {
            0.0
        };
    }

    // Compute tangents using Fritsch-Carlson method
    let mut m = vec![0.0f32; n];
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for i in 1..n - 1 {
        if delta[i - 1] * delta[i] <= 0.0 {
            m[i] = 0.0;
        } else {
            m[i] = (delta[i - 1] + delta[i]) * 0.5;
        }
    }

    // Enforce monotonicity
    for i in 0..n - 1 {
        if delta[i].abs() < 1e-10 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];
            let s = alpha * alpha + beta * beta;
            if s > 9.0 {
                let tau = 3.0 / s.sqrt();
                m[i] = tau * alpha * delta[i];
                m[i + 1] = tau * beta * delta[i];
            }
        }
    }

    // Evaluate at each LUT position
    for (idx, v) in lut.iter_mut().enumerate() {
        let x = idx as f32 / 255.0;

        // Find the interval
        let seg = pts[..n - 1]
            .iter()
            .rposition(|p| x >= p.0)
            .unwrap_or_default();

        let x0 = pts[seg].0;
        let x1 = pts[seg + 1].0;
        let y0 = pts[seg].1;
        let y1 = pts[seg + 1].1;
        let dx = x1 - x0;

        if dx.abs() < 1e-10 {
            *v = y0;
            continue;
        }

        let t = ((x - x0) / dx).clamp(0.0, 1.0);
        let t2 = t * t;
        let t3 = t2 * t;

        // Hermite basis
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        *v = (h00 * y0 + h10 * dx * m[seg] + h01 * y1 + h11 * dx * m[seg + 1]).clamp(0.0, 1.0);
    }

    lut
}

impl Filter for ToneCurve {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::L_ONLY
    }

    fn apply(&self, planes: &mut OklabPlanes, _ctx: &mut FilterContext) {
        if self.is_identity() {
            return;
        }

        for v in &mut planes.l {
            let x = (*v * 255.0).clamp(0.0, 255.0);
            let idx = x as usize;
            let frac = x - idx as f32;
            let lo = self.lut[idx.min(255)];
            let hi = self.lut[(idx + 1).min(255)];
            *v = lo + frac * (hi - lo);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_curve() {
        let mut planes = OklabPlanes::new(4, 4);
        for (i, v) in planes.l.iter_mut().enumerate() {
            *v = i as f32 / 16.0;
        }
        let orig = planes.l.clone();
        ToneCurve::default().apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, orig);
    }

    #[test]
    fn s_curve_increases_contrast() {
        let curve = ToneCurve::from_points(&[(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)]);

        let mut planes = OklabPlanes::new(4, 1);
        planes.l[0] = 0.2;
        planes.l[1] = 0.4;
        planes.l[2] = 0.6;
        planes.l[3] = 0.8;
        curve.apply(&mut planes, &mut FilterContext::new());

        // S-curve should darken shadows and brighten highlights
        assert!(planes.l[0] < 0.2, "shadows should darken: {}", planes.l[0]);
        assert!(
            planes.l[3] > 0.8,
            "highlights should brighten: {}",
            planes.l[3]
        );
    }

    #[test]
    fn output_stays_in_range() {
        let curve = ToneCurve::from_points(&[(0.0, 0.0), (0.5, 0.9), (1.0, 1.0)]);

        let mut planes = OklabPlanes::new(4, 1);
        planes.l[0] = 0.0;
        planes.l[1] = 0.5;
        planes.l[2] = 1.0;
        planes.l[3] = 1.5; // out of range input
        curve.apply(&mut planes, &mut FilterContext::new());

        for &v in &planes.l {
            assert!(v >= 0.0 && v <= 1.0, "output out of range: {v}");
        }
    }

    #[test]
    fn preserves_endpoints() {
        let curve = ToneCurve::from_points(&[(0.0, 0.0), (0.3, 0.5), (1.0, 1.0)]);
        assert!(
            (curve.lut[0] - 0.0).abs() < 1e-3,
            "black should map to black"
        );
        assert!(
            (curve.lut[255] - 1.0).abs() < 1e-3,
            "white should map to white"
        );
    }

    #[test]
    fn does_not_modify_chroma() {
        let curve = ToneCurve::from_points(&[(0.0, 0.0), (0.5, 0.7), (1.0, 1.0)]);
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.l {
            *v = 0.5;
        }
        for v in &mut planes.a {
            *v = 0.1;
        }
        let a_orig = planes.a.clone();
        curve.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.a, a_orig);
    }

    #[test]
    fn from_lut_identity() {
        // 256-entry identity LUT
        let identity: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let curve = ToneCurve::from_lut(&identity);
        assert!(
            curve.is_identity(),
            "identity LUT should produce identity curve"
        );
    }

    #[test]
    fn from_lut_gamma() {
        // Gamma 2.2 curve as a 4096-entry LUT (like Apple ProfileToneCurve)
        let gamma_lut: Vec<f32> = (0..4096)
            .map(|i| {
                let x = i as f32 / 4095.0;
                x.powf(1.0 / 2.2)
            })
            .collect();
        let curve = ToneCurve::from_lut(&gamma_lut);

        // Check midpoint: gamma(0.5) = 0.5^(1/2.2) ≈ 0.73
        let mid = curve.lut[128]; // x = 0.502
        assert!((mid - 0.73).abs() < 0.02, "gamma(0.5) ≈ 0.73, got {mid}");

        // Should brighten dark values
        assert!(
            curve.lut[64] > 64.0 / 255.0,
            "gamma should brighten shadows"
        );
    }

    #[test]
    fn from_lut_small() {
        // Tiny 3-entry LUT [0.0, 0.8, 1.0] — should resample to 256
        let curve = ToneCurve::from_lut(&[0.0, 0.8, 1.0]);
        assert!((curve.lut[0] - 0.0).abs() < 0.01);
        assert!((curve.lut[255] - 1.0).abs() < 0.01);
        // Midpoint should be ~0.8
        assert!(
            (curve.lut[128] - 0.8).abs() < 0.02,
            "mid={}",
            curve.lut[128]
        );
    }

    #[test]
    fn from_lut_preserves_chroma() {
        let gamma_lut: Vec<f32> = (0..256)
            .map(|i| {
                let x = i as f32 / 255.0;
                x.powf(0.5)
            })
            .collect();
        let curve = ToneCurve::from_lut(&gamma_lut);

        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.l {
            *v = 0.5;
        }
        for v in &mut planes.a {
            *v = 0.1;
        }
        for v in &mut planes.b {
            *v = -0.05;
        }
        let a_orig = planes.a.clone();
        let b_orig = planes.b.clone();
        curve.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.a, a_orig, "from_lut should not modify a channel");
        assert_eq!(planes.b, b_orig, "from_lut should not modify b channel");
    }

    #[test]
    fn monotone_interpolation() {
        let curve = ToneCurve::from_points(&[(0.0, 0.0), (0.3, 0.2), (0.7, 0.8), (1.0, 1.0)]);
        // Check monotonicity
        for i in 1..256 {
            assert!(
                curve.lut[i] >= curve.lut[i - 1] - 1e-5,
                "LUT not monotone at {i}: {} < {}",
                curve.lut[i],
                curve.lut[i - 1]
            );
        }
    }
}
