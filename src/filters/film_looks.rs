use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::filter::Filter;
use crate::param_schema::*;
use crate::planes::OklabPlanes;
use crate::prelude::*;

use super::cube_lut::{CubeLut, TensorLut};

/// Film look presets using compressed tensor LUTs.
///
/// Each preset is a mathematical RGB→RGB transform decomposed into a
/// rank-8 tensor approximation (~9.5 KB per look). No copyrighted LUT
/// data — all transforms are derived from first-principles color science.
///
/// Use `FilmLook::new(preset)` to create, then use as a `Filter`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FilmLook {
    tensor: TensorLut,
    /// Blend strength. 1.0 = full effect, 0.0 = bypass.
    pub strength: f32,
    preset: FilmPreset,
}

/// Available film look presets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum FilmPreset {
    /// Bleach bypass: high contrast, desaturated, gritty.
    /// Emulates skipping the bleach step in film processing.
    BleachBypass,
    /// Cross-processed: shifted color channels, saturated, punchy.
    /// Emulates developing film in the wrong chemistry.
    CrossProcess,
    /// Teal and orange: cinematic complementary color grade.
    /// Shadows pushed teal, highlights pushed warm.
    TealOrange,
    /// Faded film: lifted blacks, low contrast, muted colors.
    /// The "Instagram vintage" look done properly.
    FadedFilm,
    /// Golden hour: warm light, soft contrast, glowing highlights.
    GoldenHour,
    /// Cool chrome: slight blue-green cast, punchy contrast.
    /// Chrome slide film character.
    CoolChrome,
    /// Print film: warm shadows, soft shoulder rolloff.
    /// Motion picture print film character.
    PrintFilm,
    /// Noir: high contrast, heavy desaturation, deep blacks.
    Noir,
    /// Technicolor: vivid, saturated, slightly warm.
    /// Two-strip Technicolor-inspired color rendering.
    Technicolor,
    /// Matte: lifted blacks, reduced highlights, low saturation.
    /// Fashion/editorial look.
    Matte,
}

impl FilmPreset {
    /// All available presets.
    pub const ALL: &[FilmPreset] = &[
        FilmPreset::BleachBypass,
        FilmPreset::CrossProcess,
        FilmPreset::TealOrange,
        FilmPreset::FadedFilm,
        FilmPreset::GoldenHour,
        FilmPreset::CoolChrome,
        FilmPreset::PrintFilm,
        FilmPreset::Noir,
        FilmPreset::Technicolor,
        FilmPreset::Matte,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::BleachBypass => "Bleach Bypass",
            Self::CrossProcess => "Cross Process",
            Self::TealOrange => "Teal & Orange",
            Self::FadedFilm => "Faded Film",
            Self::GoldenHour => "Golden Hour",
            Self::CoolChrome => "Cool Chrome",
            Self::PrintFilm => "Print Film",
            Self::Noir => "Noir",
            Self::Technicolor => "Technicolor",
            Self::Matte => "Matte",
        }
    }

    /// Machine identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::BleachBypass => "bleach_bypass",
            Self::CrossProcess => "cross_process",
            Self::TealOrange => "teal_orange",
            Self::FadedFilm => "faded_film",
            Self::GoldenHour => "golden_hour",
            Self::CoolChrome => "cool_chrome",
            Self::PrintFilm => "print_film",
            Self::Noir => "noir",
            Self::Technicolor => "technicolor",
            Self::Matte => "matte",
        }
    }

    /// Look up a preset by its machine identifier.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|p| p.id() == id)
    }
}

/// LUT generation size and tensor rank for presets.
const PRESET_LUT_SIZE: usize = 17;
const PRESET_RANK: usize = 8;
const PRESET_ALS_ITERATIONS: usize = 25;

impl FilmLook {
    /// Create a film look from a preset.
    ///
    /// Generates the LUT and decomposes it on first call.
    /// The result is ~5–10 KB in memory.
    pub fn new(preset: FilmPreset) -> Self {
        let lut = generate_preset_lut(preset);
        let tensor = TensorLut::decompose(&lut, PRESET_RANK, PRESET_ALS_ITERATIONS);
        Self {
            tensor,
            strength: 1.0,
            preset,
        }
    }

    /// Create from a pre-computed TensorLut (for embedded presets).
    pub fn from_tensor(preset: FilmPreset, tensor: TensorLut) -> Self {
        Self {
            tensor,
            strength: 1.0,
            preset,
        }
    }

    /// Which preset this look uses.
    pub fn preset(&self) -> FilmPreset {
        self.preset
    }

    /// Access the underlying tensor LUT.
    pub fn tensor(&self) -> &TensorLut {
        &self.tensor
    }

    /// Storage size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.tensor.size_bytes()
    }
}

static FILM_LOOK_SCHEMA: FilterSchema = FilterSchema {
    name: "film_look",
    label: "Film Look",
    description: "Film emulation presets using compressed tensor LUTs",
    group: FilterGroup::Color,
    params: &[ParamDesc {
        name: "strength",
        label: "Strength",
        description: "Blend strength (0 = bypass, 1 = full effect)",
        kind: ParamKind::Float {
            min: 0.0,
            max: 1.0,
            default: 1.0,
            identity: 0.0,
            step: 0.05,
        },
        unit: "",
        section: "Main",
        slider: SliderMapping::Linear,
    }],
};

impl Describe for FilmLook {
    fn schema() -> &'static FilterSchema {
        &FILM_LOOK_SCHEMA
    }

    fn get_param(&self, name: &str) -> Option<ParamValue> {
        match name {
            "strength" => Some(ParamValue::Float(self.strength)),
            _ => None,
        }
    }

    fn set_param(&mut self, name: &str, value: ParamValue) -> bool {
        let v = match value.as_f32() {
            Some(v) => v,
            None => return false,
        };
        match name {
            "strength" => self.strength = v,
            _ => return false,
        }
        true
    }
}

impl Filter for FilmLook {
    fn channel_access(&self) -> ChannelAccess {
        ChannelAccess::L_AND_CHROMA
    }

    fn apply(&self, planes: &mut OklabPlanes, _ctx: &mut FilterContext) {
        if self.strength.abs() < 1e-6 {
            return;
        }

        use zenpixels_convert::oklab;
        let m1_inv = oklab::lms_to_rgb_matrix(zenpixels::ColorPrimaries::Bt709)
            .expect("BT.709 always supported");
        let m1 = oklab::rgb_to_lms_matrix(zenpixels::ColorPrimaries::Bt709)
            .expect("BT.709 always supported");

        let n = planes.pixel_count();
        let blend = self.strength.clamp(0.0, 1.0);
        let inv_blend = 1.0 - blend;

        for i in 0..n {
            let [r, g, b] = oklab::oklab_to_rgb(planes.l[i], planes.a[i], planes.b[i], &m1_inv);
            let rgb = [r.max(0.0), g.max(0.0), b.max(0.0)];
            let lut_rgb = self.tensor.lookup(rgb);

            let r2 = inv_blend * r + blend * lut_rgb[0];
            let g2 = inv_blend * g + blend * lut_rgb[1];
            let b2 = inv_blend * b + blend * lut_rgb[2];

            let [l, oa, ob] = oklab::rgb_to_oklab(r2.max(0.0), g2.max(0.0), b2.max(0.0), &m1);
            planes.l[i] = l;
            planes.a[i] = oa;
            planes.b[i] = ob;
        }
    }
}

// ── Preset LUT generators ────────────────────────────────────────────
//
// Each generates a mathematical RGB→RGB transform. All operate on
// linear [0,1] RGB. No copyrighted data.

fn generate_preset_lut(preset: FilmPreset) -> CubeLut {
    let size = PRESET_LUT_SIZE;
    let mut lut = CubeLut::identity(size);
    let scale = 1.0 / (size - 1) as f32;

    for ri in 0..size {
        for gi in 0..size {
            for bi in 0..size {
                let r = ri as f32 * scale;
                let g = gi as f32 * scale;
                let b = bi as f32 * scale;
                let idx = ri * size * size + gi * size + bi;
                lut.data_mut()[idx] = match preset {
                    FilmPreset::BleachBypass => bleach_bypass(r, g, b),
                    FilmPreset::CrossProcess => cross_process(r, g, b),
                    FilmPreset::TealOrange => teal_orange(r, g, b),
                    FilmPreset::FadedFilm => faded_film(r, g, b),
                    FilmPreset::GoldenHour => golden_hour(r, g, b),
                    FilmPreset::CoolChrome => cool_chrome(r, g, b),
                    FilmPreset::PrintFilm => print_film(r, g, b),
                    FilmPreset::Noir => noir(r, g, b),
                    FilmPreset::Technicolor => technicolor(r, g, b),
                    FilmPreset::Matte => matte(r, g, b),
                };
            }
        }
    }
    lut
}

/// BT.709 luminance.
#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// S-curve: smooth contrast boost.
#[inline]
fn s_curve(x: f32, strength: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    // Attempt smooth blend so 0 strength = identity
    let curved = if x < 0.5 {
        2.0 * x * x
    } else {
        1.0 - 2.0 * (1.0 - x) * (1.0 - x)
    };
    x + strength * (curved - x)
}

/// Inverse S-curve: reduce contrast.
#[inline]
fn inv_s_curve(x: f32, strength: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let flat = if x < 0.5 {
        (x * 0.5).sqrt()
    } else {
        1.0 - (0.5 * (1.0 - x)).sqrt()
    };
    x + strength * (flat - x)
}

/// Desaturate toward luma by a factor.
#[inline]
fn desat(r: f32, g: f32, b: f32, amount: f32) -> [f32; 3] {
    let l = luma(r, g, b);
    [
        (r + amount * (l - r)).clamp(0.0, 1.0),
        (g + amount * (l - g)).clamp(0.0, 1.0),
        (b + amount * (l - b)).clamp(0.0, 1.0),
    ]
}

/// Film shoulder rolloff.
#[inline]
fn shoulder(x: f32, knee: f32) -> f32 {
    if x < knee {
        x
    } else {
        let over = x - knee;
        let range = 1.0 - knee;
        knee + range * (1.0 - (-over / range).exp())
    }
}

fn bleach_bypass(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Strong S-curve + heavy desaturation
    let [r, g, b] = desat(r, g, b, 0.6);
    [
        s_curve(r, 0.8).clamp(0.0, 1.0),
        s_curve(g, 0.8).clamp(0.0, 1.0),
        s_curve(b, 0.8).clamp(0.0, 1.0),
    ]
}

fn cross_process(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Per-channel curve shifts: boost R highlights, suppress B shadows
    let r_out = s_curve(r, 0.4) + 0.02;
    let g_out = inv_s_curve(g, 0.2);
    let b_out = s_curve(b * 0.9, 0.3) - 0.03;
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

fn teal_orange(r: f32, g: f32, b: f32) -> [f32; 3] {
    let l = luma(r, g, b);
    // Shadow → teal (reduce R, boost B slightly), highlight → warm
    let shadow = (1.0 - l * 2.0).max(0.0); // 1 at black, 0 at mid
    let highlight = ((l - 0.5) * 2.0).max(0.0); // 0 at mid, 1 at white

    let r_out = r - shadow * 0.06 + highlight * 0.04;
    let g_out = g - shadow * 0.01 - highlight * 0.01;
    let b_out = b + shadow * 0.05 - highlight * 0.05;

    // Mild S-curve for punch
    [
        s_curve(r_out, 0.3).clamp(0.0, 1.0),
        s_curve(g_out, 0.3).clamp(0.0, 1.0),
        s_curve(b_out, 0.3).clamp(0.0, 1.0),
    ]
}

fn faded_film(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Lift blacks, lower whites, desaturate
    let lift = 0.06;
    let ceil = 0.94;
    let range = ceil - lift;
    let r_out = lift + r * range;
    let g_out = lift + g * range;
    let b_out = lift + b * range;
    let [r_out, g_out, b_out] = desat(r_out, g_out, b_out, 0.3);
    // Slight warm shift in shadows
    let l = luma(r, g, b);
    let shadow = (1.0 - l * 3.0).max(0.0);
    [
        (r_out + shadow * 0.02).clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        (b_out - shadow * 0.01).clamp(0.0, 1.0),
    ]
}

fn golden_hour(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Warm shift, lifted shadows, soft shoulder
    let r_out = shoulder(r * 1.05 + 0.03, 0.85);
    let g_out = shoulder(g * 1.0 + 0.01, 0.88);
    let b_out = shoulder(b * 0.88, 0.9);
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

fn cool_chrome(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Slight blue-green cast, punchy contrast
    let r_out = s_curve(r * 0.98, 0.4);
    let g_out = s_curve(g * 1.0 + 0.01, 0.3);
    let b_out = s_curve(b * 1.04 + 0.02, 0.3);
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

fn print_film(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Warm shadows, soft shoulder, slight desaturation in highlights
    let l = luma(r, g, b);
    let shadow = (1.0 - l * 2.5).max(0.0);
    let r_out = shoulder(r + shadow * 0.04, 0.82);
    let g_out = shoulder(g + shadow * 0.01, 0.85);
    let b_out = shoulder(b - shadow * 0.02, 0.88);
    // Desaturate highlights
    let highlight = ((l - 0.6) * 2.5).max(0.0).min(1.0);
    let [r_out, g_out, b_out] = desat(r_out, g_out, b_out, highlight * 0.2);
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

fn noir(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Heavy desat, strong S-curve, crush blacks
    let [r, g, b] = desat(r, g, b, 0.85);
    let r_out = s_curve((r - 0.02).max(0.0), 0.7);
    let g_out = s_curve((g - 0.02).max(0.0), 0.7);
    let b_out = s_curve((b - 0.02).max(0.0), 0.7);
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

fn technicolor(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Vivid, saturated, slightly warm. Inspired by two-strip process:
    // boost reds and cyans, compress greens
    let r_out = r * 1.08 + 0.01;
    let g_out = g * 0.95;
    let b_out = b * 1.04;
    // Slight S-curve for punch
    [
        s_curve(r_out, 0.3).clamp(0.0, 1.0),
        s_curve(g_out, 0.2).clamp(0.0, 1.0),
        s_curve(b_out, 0.25).clamp(0.0, 1.0),
    ]
}

fn matte(r: f32, g: f32, b: f32) -> [f32; 3] {
    // Lifted blacks, lowered highlights, low saturation
    let lift = 0.08;
    let ceil = 0.90;
    let range = ceil - lift;
    let r_out = lift + inv_s_curve(r, 0.2) * range;
    let g_out = lift + inv_s_curve(g, 0.2) * range;
    let b_out = lift + inv_s_curve(b, 0.2) * range;
    let [r_out, g_out, b_out] = desat(r_out, g_out, b_out, 0.25);
    [
        r_out.clamp(0.0, 1.0),
        g_out.clamp(0.0, 1.0),
        b_out.clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;

    #[test]
    fn all_presets_build() {
        for &preset in FilmPreset::ALL {
            let look = FilmLook::new(preset);
            assert!(look.size_bytes() > 0, "{}: zero size", preset.name());
            std::eprintln!(
                "{:15} {:>6} bytes  (rank {}, grid {})",
                preset.name(),
                look.size_bytes(),
                PRESET_RANK,
                look.tensor().grid_size(),
            );
        }
    }

    #[test]
    fn all_presets_accuracy() {
        for &preset in FilmPreset::ALL {
            let lut = generate_preset_lut(preset);
            let look = FilmLook::new(preset);
            let acc = lut.measure_accuracy(&|rgb| look.tensor().lookup(rgb), 33);
            let max_8bit = (acc.max_diff * 255.0).ceil() as u32;
            std::eprintln!(
                "{:15} max={:.4} ({:>2}@8bit) avg={:.6}",
                preset.name(),
                acc.max_diff,
                max_8bit,
                acc.avg_diff,
            );
            assert!(
                acc.max_diff < 0.1,
                "{}: max_diff too high: {}",
                preset.name(),
                acc.max_diff
            );
        }
    }

    #[test]
    fn strength_zero_is_bypass() {
        let mut look = FilmLook::new(FilmPreset::BleachBypass);
        look.strength = 0.0;

        let mut planes = OklabPlanes::new(4, 4);
        for (i, v) in planes.l.iter_mut().enumerate() {
            *v = 0.3 + (i as f32) * 0.01;
        }
        for v in &mut planes.a {
            *v = 0.05;
        }
        let l_orig = planes.l.clone();
        look.apply(&mut planes, &mut FilterContext::new());
        assert_eq!(planes.l, l_orig);
    }

    #[test]
    fn preset_from_id_roundtrip() {
        for &preset in FilmPreset::ALL {
            let id = preset.id();
            let back = FilmPreset::from_id(id).unwrap();
            assert_eq!(back, preset);
        }
    }

    #[test]
    fn tensor_serialization_roundtrip() {
        let look = FilmLook::new(FilmPreset::TealOrange);
        let bytes = look.tensor().to_bytes();
        let restored = TensorLut::from_bytes(&bytes).unwrap();
        // Spot-check a few values
        let test_pts = [[0.5, 0.3, 0.7], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        for pt in &test_pts {
            let a = look.tensor().lookup(*pt);
            let b = restored.lookup(*pt);
            for ch in 0..3 {
                assert!(
                    (a[ch] - b[ch]).abs() < 1e-6,
                    "serialization mismatch at {pt:?} ch{ch}: {} vs {}",
                    a[ch],
                    b[ch]
                );
            }
        }
    }

    #[test]
    fn total_embedded_size() {
        let mut total = 0;
        for &preset in FilmPreset::ALL {
            let look = FilmLook::new(preset);
            total += look.size_bytes();
        }
        std::eprintln!(
            "Total embedded size for {} presets: {} bytes ({:.1} KB)",
            FilmPreset::ALL.len(),
            total,
            total as f64 / 1024.0,
        );
        // Should be well under 100 KB total
        assert!(total < 100_000, "presets too large: {} bytes", total);
    }
}
