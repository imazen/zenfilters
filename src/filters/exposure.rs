use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::planes::OklabPlanes;
use crate::simd;

/// Exposure adjustment — simulates changing light intensity by ±stops.
///
/// +1 stop doubles linear light, -1 halves it. Because Oklab uses a
/// cube-root transform, scaling linear light by `f` means scaling all
/// Oklab channels (L, a, b) by `f^(1/3)`. This preserves hue and
/// saturation exactly, unlike scaling L alone (which desaturates).
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct Exposure {
    /// Exposure adjustment in stops. 0.0 = no change.
    pub stops: f32,
}

impl Filter for Exposure {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::L_AND_CHROMA
    }

    fn tag(&self) -> crate::filter_compat::FilterTag {
        crate::filter_compat::FilterTag::Exposure
    }
    fn apply(&self, planes: &mut OklabPlanes, _ctx: &mut FilterContext) {
        // Linear light factor = 2^stops.
        // Oklab factor = (2^stops)^(1/3) = 2^(stops/3) because Oklab
        // channels are linear functions of cube-root LMS values.
        let factor = 2.0f32.powf(self.stops / 3.0);
        simd::scale_plane(&mut planes.l, factor);
        simd::scale_plane(&mut planes.a, factor);
        simd::scale_plane(&mut planes.b, factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_stops_is_identity() {
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.l {
            *v = 0.5;
        }
        let original = planes.l.clone();
        Exposure { stops: 0.0 }.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, original);
    }

    #[test]
    fn positive_stops_brighten() {
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.l {
            *v = 0.3;
        }
        Exposure { stops: 1.0 }.apply(&mut planes, &mut FilterContext::new());
        // +1 stop = 2x linear light = 2^(1/3) ≈ 1.2599 in Oklab
        let expected = 0.3 * 2.0f32.powf(1.0 / 3.0);
        for &v in &planes.l {
            assert!(
                (v - expected).abs() < 1e-5,
                "expected ~{expected:.4}, got {v}"
            );
        }
    }

    #[test]
    fn scales_chroma_proportionally() {
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
        Exposure { stops: 1.0 }.apply(&mut planes, &mut FilterContext::new());

        // All channels scale by the same factor
        let factor = 2.0f32.powf(1.0 / 3.0);
        for &v in &planes.a {
            assert!((v - 0.1 * factor).abs() < 1e-5);
        }
        for &v in &planes.b {
            assert!((v - (-0.05 * factor)).abs() < 1e-5);
        }
    }
}
