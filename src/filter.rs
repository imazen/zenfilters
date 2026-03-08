use crate::access::ChannelAccess;
use crate::context::FilterContext;
use crate::planes::OklabPlanes;

/// A photo filter that operates on planar Oklab f32 data.
///
/// Filters modify `OklabPlanes` in-place. The pipeline guarantees that
/// planes are in the correct format (f32 Oklab) before calling `apply`.
///
/// Filters are infallible — any validation (e.g., parameter clamping)
/// happens at construction time, not at apply time.
///
/// The `ctx` parameter provides a pool of reusable scratch buffers.
/// Neighborhood filters should use `ctx.take_f32()` and `ctx.return_f32()`
/// for temporary planes instead of allocating fresh vectors each call.
pub trait Filter: Send + Sync {
    /// Which planes this filter reads and writes.
    fn channel_access(&self) -> ChannelAccess;

    /// Whether this filter needs neighborhood access (reads adjacent pixels).
    ///
    /// Per-pixel filters return false. Neighborhood filters (clarity,
    /// brilliance, bilateral) return true.
    fn is_neighborhood(&self) -> bool {
        false
    }

    /// Apply the filter in-place to the given planes.
    ///
    /// `ctx` provides reusable scratch buffers — neighborhood filters should
    /// borrow temporary planes from `ctx` instead of allocating.
    fn apply(&self, planes: &mut OklabPlanes, ctx: &mut FilterContext);
}
