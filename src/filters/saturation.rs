use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::param_schema::*;
use crate::planes::OklabPlanes;
use crate::simd;

/// Uniform saturation adjustment on Oklab a/b channels.
///
/// Scales chroma (a, b) by a constant factor. 1.0 = no change,
/// 0.0 = grayscale, 2.0 = double saturation.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct Saturation {
    /// Saturation factor. 1.0 = no change, 0.0 = grayscale, 2.0 = double.
    ///
    /// For slider integration, use [`Saturation::from_slider`] which maps
    /// a 0.0–1.0 range with 0.5 as the identity point (no change).
    pub factor: f32,
}

impl Saturation {
    /// Create from a 0.0–1.0 slider where 0.5 = identity (no change).
    ///
    /// - Slider 0.0 → factor 0.0 (grayscale)
    /// - Slider 0.5 → factor 1.0 (no change)
    /// - Slider 1.0 → factor 2.0 (double saturation)
    pub fn from_slider(slider: f32) -> Self {
        Self {
            factor: crate::slider::saturation_from_slider(slider.clamp(0.0, 1.0)),
        }
    }
}

impl Default for Saturation {
    fn default() -> Self {
        Self { factor: 1.0 }
    }
}

impl Filter for Saturation {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::CHROMA_ONLY
    }

    fn tag(&self) -> crate::filter_compat::FilterTag {
        crate::filter_compat::FilterTag::Saturation
    }
    fn apply(&self, planes: &mut OklabPlanes, ctx: &mut FilterContext) {
        if (self.factor - 1.0).abs() < 1e-6 {
            return;
        }

        if ctx.working_space == crate::pipeline::WorkingSpace::Srgb {
            // sRGB mode: ImageMagick-compatible saturation via HSL modulate.
            // IM converts to HSL, scales S by factor, converts back.
            // Equivalent matrix form using Rec.601 NTSC luma weights:
            let s = self.factor;
            let n = planes.l.len();
            for i in 0..n {
                let r = planes.l[i];
                let g = planes.a[i];
                let b = planes.b[i];
                // Convert to HSL, scale S, convert back
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                let l = (max + min) * 0.5;
                let delta = max - min;
                if delta < 1e-6 {
                    continue; // achromatic — no change
                }
                let sat = if l <= 0.5 {
                    delta / (max + min)
                } else {
                    delta / (2.0 - max - min)
                };
                let new_sat = (sat * s).clamp(0.0, 1.0);

                // Hue (in 0-6)
                let hue = if (max - r).abs() < 1e-6 {
                    (g - b) / delta + (if g < b { 6.0 } else { 0.0 })
                } else if (max - g).abs() < 1e-6 {
                    (b - r) / delta + 2.0
                } else {
                    (r - g) / delta + 4.0
                };

                // HSL to RGB
                let c = (1.0 - (2.0 * l - 1.0).abs()) * new_sat;
                let x = c * (1.0 - ((hue % 2.0) - 1.0).abs());
                let m = l - c * 0.5;
                let (r1, g1, b1) = match hue as u32 {
                    0 => (c, x, 0.0),
                    1 => (x, c, 0.0),
                    2 => (0.0, c, x),
                    3 => (0.0, x, c),
                    4 => (x, 0.0, c),
                    _ => (c, 0.0, x),
                };
                planes.l[i] = (r1 + m).clamp(0.0, 1.0);
                planes.a[i] = (g1 + m).clamp(0.0, 1.0);
                planes.b[i] = (b1 + m).clamp(0.0, 1.0);
            }
            return;
        }

        // Oklab: scale chroma directly
        simd::scale_plane(&mut planes.a, self.factor);
        simd::scale_plane(&mut planes.b, self.factor);
    }
}

static SATURATION_SCHEMA: FilterSchema = FilterSchema {
    name: "saturation",
    label: "Saturation",
    description: "Uniform chroma scaling on Oklab a/b channels",
    group: FilterGroup::Color,
    params: &[ParamDesc {
        name: "factor",
        label: "Saturation",
        description: "Saturation multiplier (0 = grayscale, 1 = unchanged, 2 = double)",
        kind: ParamKind::Float {
            min: 0.0,
            max: 2.0,
            default: 1.0,
            identity: 1.0,
            step: 0.05,
        },
        unit: "×",
        section: "Main",
        slider: SliderMapping::FactorCentered,
    }],
};

impl Describe for Saturation {
    fn schema() -> &'static FilterSchema {
        &SATURATION_SCHEMA
    }

    fn get_param(&self, name: &str) -> Option<ParamValue> {
        match name {
            "factor" => Some(ParamValue::Float(self.factor)),
            _ => None,
        }
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> bool {
        let v = match value.as_f32() {
            Some(v) => v,
            None => return false,
        };
        match name {
            "factor" => self.factor = v,
            _ => return false,
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_is_identity() {
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.a {
            *v = 0.1;
        }
        let original = planes.a.clone();
        Saturation { factor: 1.0 }.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.a, original);
    }

    #[test]
    fn zero_is_grayscale() {
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.a {
            *v = 0.1;
        }
        for v in &mut planes.b {
            *v = -0.05;
        }
        Saturation { factor: 0.0 }.apply(&mut planes, &mut FilterContext::new());
        for &v in &planes.a {
            assert!(v.abs() < 1e-6);
        }
        for &v in &planes.b {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn does_not_modify_l() {
        let mut planes = OklabPlanes::new(4, 4);
        for v in &mut planes.l {
            *v = 0.5;
        }
        let original = planes.l.clone();
        Saturation { factor: 2.0 }.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, original);
    }
}
