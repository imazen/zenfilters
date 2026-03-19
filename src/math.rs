//! no_std-compatible scalar f32 math via libm.
//!
//! All scalar transcendental functions used in filters are routed through
//! this module so the crate compiles without std. These are thin wrappers
//! around `libm` functions.

#![allow(dead_code)]

#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[inline(always)]
pub fn expf(x: f32) -> f32 {
    libm::expf(x)
}

#[inline(always)]
pub fn logf(x: f32) -> f32 {
    libm::logf(x)
}

#[inline(always)]
pub fn log2f(x: f32) -> f32 {
    libm::log2f(x)
}

#[inline(always)]
pub fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline(always)]
pub fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

#[inline(always)]
pub fn atan2f(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

#[inline(always)]
pub fn cbrtf(x: f32) -> f32 {
    libm::cbrtf(x)
}

#[inline(always)]
pub fn fabsf(x: f32) -> f32 {
    libm::fabsf(x)
}

// f64 variants for high-precision statistics
#[inline(always)]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

#[inline(always)]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[inline(always)]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[inline(always)]
pub fn log_gamma(x: f64) -> f64 {
    libm::lgamma_r(x).0
}
