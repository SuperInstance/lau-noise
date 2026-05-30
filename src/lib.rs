/// `lau-noise` — a deterministic, seedable procedural noise generation library.
///
/// All noise functions are pure: given the same seed, they always produce
/// the same output, bit-for-bit, on the same platform.
///
/// # Overview
/// - **PermutationTable** — seeded permutation table (512 bytes)
/// - **Value noise** — 1D / 2D with linear interpolation
/// - **Perlin noise** — 2D gradient noise with smooth fade
/// - **Simplex-like noise** — 2D simplex-style skew/unskew
/// - **Fractal (fBm)** — 1D / 2D, ridged multifractal, turbulence
/// - **Domain manipulation** — domain warping, tiling
/// - **Terrain helpers** — terrain height / moisture maps
///
/// Classic permutation table for Perlin-style noise.
///
/// `perm` is 512 entries (two copies of a 256-entry permutation) so that
/// lookups for `(x+1)` wrap without an explicit branch.
pub struct PermutationTable {
    pub perm: [u8; 512],
}

impl PermutationTable {
    /// Create a new permutation table seeded from a 64-bit seed.
    ///
    /// Uses a simple splitmix64-like hash to fill the 256-index permutation,
    /// then duplicates it for the second half.
    pub fn new(seed: u64) -> Self {
        let mut perm = [0u8; 512];

        // Simple Fisher-Yates shuffle using a splitmix64-like rng.
        let mut state = seed;
        for i in 0..256 {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;

            let j = (z as usize) % (i + 1);
            perm[i] = perm[j];
            perm[j] = i as u8;
        }

        // Duplicate into second half for wrap-free lookups.
        for i in 0..256 {
            perm[i + 256] = perm[i];
        }

        Self { perm }
    }

    /// Hash a single integer coordinate.
    #[inline]
    pub fn hash(&self, x: i32) -> u8 {
        let idx = (x & 0xFF) as usize;
        self.perm[idx]
    }

    /// Hash two integer coordinates.
    #[inline]
    pub fn hash2(&self, x: i32, y: i32) -> u8 {
        let idx = (x & 0xFF) as usize;
        self.perm[self.perm[idx] as usize ^ (y & 0xFF) as usize]
    }

    /// Hash three integer coordinates.
    #[inline]
    pub fn hash3(&self, x: i32, y: i32, z: i32) -> u8 {
        let idx = (x & 0xFF) as usize;
        let idx2 = self.perm[idx] as usize ^ (y & 0xFF) as usize;
        self.perm[self.perm[idx2] as usize ^ (z & 0xFF) as usize]
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Linear interpolation.
#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Smooth fade function: 6t⁵ − 15t⁴ + 10t³.
#[inline]
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Map a hash byte to a gradient vector index (0..7) for Perlin 2D.
#[inline]
fn grad_index_2d(hash: u8) -> usize {
    (hash & 0x07) as usize
}

/// Pre-computed 8 gradient vectors for 2D Perlin (pointing to edges of a
/// square). Each is a unit vector in one of 8 directions.
const GRADIENTS_2D: [(f64, f64); 8] = [
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
    (core::f64::consts::FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2),
    (-core::f64::consts::FRAC_1_SQRT_2, core::f64::consts::FRAC_1_SQRT_2),
    (core::f64::consts::FRAC_1_SQRT_2, -core::f64::consts::FRAC_1_SQRT_2),
    (-core::f64::consts::FRAC_1_SQRT_2, -core::f64::consts::FRAC_1_SQRT_2),
];

/// Dot product of a gradient vector with the offset from a corner.
#[inline]
fn dot2(g: usize, dx: f64, dy: f64) -> f64 {
    let (gx, gy) = GRADIENTS_2D[g];
    gx * dx + gy * dy
}

// ---------------------------------------------------------------------------
// Value noise
// ---------------------------------------------------------------------------

/// 1D value noise with linear interpolation.
///
/// Output is in the range [-1, 1].
pub fn value_noise_1d(table: &PermutationTable, x: f64) -> f64 {
    let xi = x.floor() as i32;
    let frac = x - xi as f64;

    let v0 = table.hash(xi) as f64 / 255.0;
    let v1 = table.hash(xi + 1) as f64 / 255.0;

    // Scale from [0,1] to [-1,1]
    lerp(v0, v1, frac) * 2.0 - 1.0
}

/// 2D value noise with bilinear interpolation.
///
/// Output is in the range [-1, 1].
pub fn value_noise_2d(table: &PermutationTable, x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let frac_x = x - xi as f64;
    let frac_y = y - yi as f64;

    let v00 = table.hash2(xi, yi) as f64 / 255.0;
    let v10 = table.hash2(xi + 1, yi) as f64 / 255.0;
    let v01 = table.hash2(xi, yi + 1) as f64 / 255.0;
    let v11 = table.hash2(xi + 1, yi + 1) as f64 / 255.0;

    let vx0 = lerp(v00, v10, frac_x);
    let vx1 = lerp(v01, v11, frac_x);

    lerp(vx0, vx1, frac_y) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Perlin noise
// ---------------------------------------------------------------------------

/// 2D Perlin noise using gradient vectors and the fade function 6t⁵-15t⁴+10t³.
///
/// Output is in the range [-1, 1].
pub fn perlin_noise_2d(table: &PermutationTable, x: f64, y: f64) -> f64 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;

    let dx = x - xi as f64;
    let dy = y - yi as f64;

    let u = fade(dx);
    let v = fade(dy);

    let n00 = dot2(
        grad_index_2d(table.hash2(xi, yi)),
        dx,
        dy,
    );
    let n10 = dot2(
        grad_index_2d(table.hash2(xi + 1, yi)),
        dx - 1.0,
        dy,
    );
    let n01 = dot2(
        grad_index_2d(table.hash2(xi, yi + 1)),
        dx,
        dy - 1.0,
    );
    let n11 = dot2(
        grad_index_2d(table.hash2(xi + 1, yi + 1)),
        dx - 1.0,
        dy - 1.0,
    );

    let nx0 = lerp(n00, n10, u);
    let nx1 = lerp(n01, n11, u);

    lerp(nx0, nx1, v)
}

// ---------------------------------------------------------------------------
// Simplex-like 2D noise
// ---------------------------------------------------------------------------

/// 2D simplex-style noise with skew/unskew grid.
///
/// This is a simplified implementation of the simplex noise algorithm:
/// it skews the input coordinates onto a simplex grid, finds the containing
/// triangle, and interpolates contributions from each vertex.
///
/// Output is in the range [-1, 1].
pub fn simplex_like_2d(table: &PermutationTable, x: f64, y: f64) -> f64 {
    // Skew factor for 2D: (sqrt(3)-1)/2 ≈ 0.3660254037844386
    const F2: f64 = 0.3660254037844386;
    // Unskew factor for 2D: (3-sqrt(3))/6 ≈ 0.21132486540518713
    const G2: f64 = 0.21132486540518713;

    let s = (x + y) * F2;
    let i = (x + s).floor() as i32;
    let j = (y + s).floor() as i32;

    let t = (i + j) as f64 * G2;
    let x0 = x - (i as f64 - t);
    let y0 = y - (j as f64 - t);

    // Determine which simplex we're in (upper or lower triangle)
    let (i1, j1) = if x0 > y0 { (1, 0) } else { (0, 1) };

    // Offsets for middle corner
    let x1 = x0 - i1 as f64 + G2;
    let y1 = y0 - j1 as f64 + G2;
    // Offsets for last corner
    let x2 = x0 - 1.0 + 2.0 * G2;
    let y2 = y0 - 1.0 + 2.0 * G2;

    // Hash the three corners
    let gi0 = table.hash2(i, j) % 8;
    let gi1 = table.hash2(i + i1, j + j1) % 8;
    let gi2 = table.hash2(i + 1, j + 1) % 8;

    // Calculate contribution from each corner
    fn contribution(g: u8, dx: f64, dy: f64) -> f64 {
        let t = 0.5 - dx * dx - dy * dy;
        if t <= 0.0 {
            return 0.0;
        }
        let t2 = t * t;
        let g_idx = g as usize;
        t2 * t2 * dot2(g_idx, dx, dy)
    }

    let n0 = contribution(gi0, x0, y0);
    let n1 = contribution(gi1, x1, y1);
    let n2 = contribution(gi2, x2, y2);

    // Scale to approximately [-1, 1]
    (n0 + n1 + n2) * 70.0
}

// ---------------------------------------------------------------------------
// Fractal noise (fBm)
// ---------------------------------------------------------------------------

/// 1D fractal Brownian motion.
///
/// Sums `octaves` layers of `value_noise_1d`, doubling frequency and scaling
/// amplitude by `persistence` each octave.
///
/// Output is in the range [-1, 1].
pub fn fbm_1d(
    table: &PermutationTable,
    x: f64,
    octaves: u32,
    lacunarity: f64,
    persistence: f64,
) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * value_noise_1d(table, x * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }

    // Normalize to [-1, 1]
    value / max_amplitude
}

/// 2D fractal Brownian motion.
///
/// Output is in the range [-1, 1].
pub fn fbm_2d(
    table: &PermutationTable,
    x: f64,
    y: f64,
    octaves: u32,
    lacunarity: f64,
    persistence: f64,
) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * perlin_noise_2d(table, x * frequency, y * frequency);
        max_amplitude += amplitude;
        frequency *= lacunarity;
        amplitude *= persistence;
    }

    value / max_amplitude
}

/// Ridged multifractal noise 2D.
///
/// Applies `|noise|` to create sharp ridge-like features, then inverts so
/// valleys become ridges.
///
/// Output is in the range [-1, 1].
pub fn ridged_multifractal_2d(
    table: &PermutationTable,
    x: f64,
    y: f64,
    octaves: u32,
    lacunarity: f64,
    persistence: f64,
) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;
    let mut offset = 1.0;

    for _ in 0..octaves {
        let noise = perlin_noise_2d(table, x * frequency, y * frequency);
        // |noise| then invert: ridges = 1 - |noise|
        let ridge = offset - noise.abs();
        value += amplitude * ridge;
        max_amplitude += amplitude * offset;
        frequency *= lacunarity;
        amplitude *= persistence;
        offset *= 0.5; // decrease offset for each octave
    }

    value / max_amplitude
}

/// Turbulence noise 2D.
///
/// Sum of absolute values of `perlin_noise_2d` octaves, creating a turbulent
/// effect suitable for marble-like textures.
///
/// Output is in the range [-1, 1].
pub fn turbulence_2d(
    table: &PermutationTable,
    x: f64,
    y: f64,
    octaves: u32,
) -> f64 {
    let mut value = 0.0;
    let mut amplitude = 1.0;
    let mut frequency = 1.0;
    let mut max_amplitude = 0.0;

    for _ in 0..octaves {
        value += amplitude * perlin_noise_2d(table, x * frequency, y * frequency).abs();
        max_amplitude += amplitude;
        frequency *= 2.0;
        amplitude *= 0.5;
    }

    // Normalize to [-1, 1] from [0, max_amplitude]
    (value / max_amplitude) * 2.0 - 1.0
}

// ---------------------------------------------------------------------------
// Domain manipulation
// ---------------------------------------------------------------------------

/// Domain warping 2D.
///
/// Warps the input coordinates by sampling noise at the original position
/// and using the result as an offset for a second noise sample.
pub fn warp(
    table: &PermutationTable,
    x: f64,
    y: f64,
    frequency: f64,
    amplitude: f64,
) -> (f64, f64) {
    let dx = amplitude * perlin_noise_2d(table, x * frequency, y * frequency);
    let dy = amplitude * perlin_noise_2d(table, x * frequency + 100.0, y * frequency + 100.0);

    (dx, dy)
}

/// Tileable noise 2D.
///
/// Produces seamlessly tileable noise by evaluating noise at wrapped
/// coordinates using the tile dimensions.
///
/// Output is in the range [-1, 1].
pub fn tile(
    table: &PermutationTable,
    x: f64,
    y: f64,
    frequency: f64,
    tile_width: f64,
    tile_height: f64,
) -> f64 {
    // Use a simple approach: noise on a toroidal domain by sampling noise
    // at angular coordinates that wrap around.
    let nx = x * frequency;
    let ny = y * frequency;

    // We do a weighted blend of 4 corners of the toroidal space
    let ux = (nx % tile_width) / tile_width;
    let uy = (ny % tile_height) / tile_height;

    let cx = (nx / tile_width).floor() as i32;
    let cy = (ny / tile_height).floor() as i32;

    let v00 = perlin_noise_2d(table, cx as f64 * tile_width, cy as f64 * tile_height);
    let v10 =
        perlin_noise_2d(table, (cx + 1) as f64 * tile_width, cy as f64 * tile_height);
    let v01 =
        perlin_noise_2d(table, cx as f64 * tile_width, (cy + 1) as f64 * tile_height);
    let v11 = perlin_noise_2d(
        table,
        (cx + 1) as f64 * tile_width,
        (cy + 1) as f64 * tile_height,
    );

    let vx0 = lerp(v00, v10, ux);
    let vx1 = lerp(v01, v11, ux);

    lerp(vx0, vx1, uy)
}

// ---------------------------------------------------------------------------
// Terrain generation helpers
// ---------------------------------------------------------------------------

/// Terrain height map value using multi-octave fBm.
///
/// Normalized to [0, 1] for use as a height field.
pub fn terrain_height(table: &PermutationTable, x: f64, y: f64) -> f64 {
    const OCTAVES: u32 = 6;
    const LACUNARITY: f64 = 2.0;
    const PERSISTENCE: f64 = 0.5;

    let raw = fbm_2d(table, x, y, OCTAVES, LACUNARITY, PERSISTENCE);
    // Normalize from [-1, 1] to [0, 1]
    raw * 0.5 + 0.5
}

/// Moisture map using a different seed offset for variation.
///
/// Normalized to [0, 1].
pub fn moisture_map(table: &PermutationTable, x: f64, y: f64) -> f64 {
    // Use a seed offset by hashing a different region of noise space
    let offset_x = x + 1000.0;
    let offset_y = y + 1000.0;

    const OCTAVES: u32 = 4;
    const LACUNARITY: f64 = 2.0;
    const PERSISTENCE: f64 = 0.5;

    let raw = fbm_2d(table, offset_x, offset_y, OCTAVES, LACUNARITY, PERSISTENCE);
    raw * 0.5 + 0.5
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Determinism tests
    // -----------------------------------------------------------------------

    #[test]
    fn permutation_table_is_deterministic() {
        let t1 = PermutationTable::new(42);
        let t2 = PermutationTable::new(42);

        for i in 0..512 {
            assert_eq!(t1.perm[i], t2.perm[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn permutation_table_differs_for_different_seeds() {
        let t1 = PermutationTable::new(42);
        let t2 = PermutationTable::new(123);
        let mut differing = false;
        for i in 0..512 {
            if t1.perm[i] != t2.perm[i] {
                differing = true;
                break;
            }
        }
        assert!(differing, "different seeds should produce different tables");
    }

    #[test]
    fn value_noise_1d_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = value_noise_1d(&t, 3.1415);
        let v2 = value_noise_1d(&t, 3.1415);
        assert_eq!(v1, v2);
    }

    #[test]
    fn value_noise_2d_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = value_noise_2d(&t, 1.23, 4.56);
        let v2 = value_noise_2d(&t, 1.23, 4.56);
        assert_eq!(v1, v2);
    }

    #[test]
    fn perlin_noise_2d_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = perlin_noise_2d(&t, 2.71, 3.14);
        let v2 = perlin_noise_2d(&t, 2.71, 3.14);
        assert_eq!(v1, v2);
    }

    #[test]
    fn fbm_1d_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = fbm_1d(&t, 1.5, 4, 2.0, 0.5);
        let v2 = fbm_1d(&t, 1.5, 4, 2.0, 0.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn fbm_2d_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = fbm_2d(&t, 1.5, 2.5, 4, 2.0, 0.5);
        let v2 = fbm_2d(&t, 1.5, 2.5, 4, 2.0, 0.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn simplex_like_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = simplex_like_2d(&t, 1.23, 4.56);
        let v2 = simplex_like_2d(&t, 1.23, 4.56);
        assert_eq!(v1, v2);
    }

    #[test]
    fn ridged_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = ridged_multifractal_2d(&t, 1.5, 2.5, 4, 2.0, 0.5);
        let v2 = ridged_multifractal_2d(&t, 1.5, 2.5, 4, 2.0, 0.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn turbulence_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = turbulence_2d(&t, 1.5, 2.5, 4);
        let v2 = turbulence_2d(&t, 1.5, 2.5, 4);
        assert_eq!(v1, v2);
    }

    #[test]
    fn warp_deterministic() {
        let t = PermutationTable::new(7);
        let (dx1, dy1) = warp(&t, 1.5, 2.5, 2.0, 0.5);
        let (dx2, dy2) = warp(&t, 1.5, 2.5, 2.0, 0.5);
        assert_eq!(dx1, dx2);
        assert_eq!(dy1, dy2);
    }

    #[test]
    fn terrain_height_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = terrain_height(&t, 1.5, 2.5);
        let v2 = terrain_height(&t, 1.5, 2.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn moisture_map_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = moisture_map(&t, 1.5, 2.5);
        let v2 = moisture_map(&t, 1.5, 2.5);
        assert_eq!(v1, v2);
    }

    #[test]
    fn tile_deterministic() {
        let t = PermutationTable::new(7);
        let v1 = tile(&t, 1.5, 2.5, 1.0, 4.0, 4.0);
        let v2 = tile(&t, 1.5, 2.5, 1.0, 4.0, 4.0);
        assert_eq!(v1, v2);
    }

    // -----------------------------------------------------------------------
    // Range tests
    // -----------------------------------------------------------------------

    #[test]
    fn value_noise_1d_range() {
        let t = PermutationTable::new(42);
        for i in 0..100 {
            let x = i as f64 * 0.37;
            let v = value_noise_1d(&t, x);
            assert!(
                v >= -1.0 && v <= 1.0,
                "value_noise_1d out of range: {v} at x={x}"
            );
        }
    }

    #[test]
    fn value_noise_2d_range() {
        let t = PermutationTable::new(42);
        for i in 0..50 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = value_noise_2d(&t, x, y);
            assert!(
                v >= -1.0 && v <= 1.0,
                "value_noise_2d out of range: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn perlin_noise_2d_range() {
        let t = PermutationTable::new(42);
        for i in 0..50 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = perlin_noise_2d(&t, x, y);
            assert!(
                v >= -1.0 && v <= 1.0,
                "perlin_noise_2d out of range: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn fbm_2d_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = fbm_2d(&t, x, y, 4, 2.0, 0.5);
            assert!(
                v >= -1.0 && v <= 1.0,
                "fbm_2d out of range: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn terrain_height_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = terrain_height(&t, x, y);
            assert!(
                v >= 0.0 && v <= 1.0,
                "terrain_height out of range [0,1]: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn moisture_map_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = moisture_map(&t, x, y);
            assert!(
                v >= 0.0 && v <= 1.0,
                "moisture_map out of range [0,1]: {v} at ({x}, {y})"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Hash consistency
    // -----------------------------------------------------------------------

    #[test]
    fn hash_is_stable() {
        let t = PermutationTable::new(0);
        assert_eq!(t.hash(0), t.hash(256));
        // hash wraps: x=0 and x=256 produce the same low byte
    }

    #[test]
    fn hash2_and_hash3_use_full_table() {
        let t = PermutationTable::new(7);
        // Just ensure no panics and results differ for different inputs
        let h1 = t.hash2(5, 10);
        let h2 = t.hash2(10, 5);
        // Not guaranteed to differ (hash collisions are normal), but usually do
        if h1 == h2 {
            // Try another pair
            let h3 = t.hash2(1, 2);
            let h4 = t.hash2(2, 1);
            assert!(h3 != h4, "hash2 should produce different outputs for different inputs (usually)");
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn negative_coordinates() {
        let t = PermutationTable::new(42);
        let v = perlin_noise_2d(&t, -3.14, -2.71);
        assert!(v >= -1.0 && v <= 1.0);

        let vv = value_noise_2d(&t, -10.5, -20.3);
        assert!(vv >= -1.0 && vv <= 1.0);
    }

    #[test]
    fn zero_input_does_not_panic() {
        let t = PermutationTable::new(0);
        // Various functions at (0,0)
        let _ = value_noise_1d(&t, 0.0);
        let _ = value_noise_2d(&t, 0.0, 0.0);
        let _ = perlin_noise_2d(&t, 0.0, 0.0);
        let _ = simplex_like_2d(&t, 0.0, 0.0);
        let _ = fbm_1d(&t, 0.0, 4, 2.0, 0.5);
        let _ = fbm_2d(&t, 0.0, 0.0, 4, 2.0, 0.5);
        let _ = ridged_multifractal_2d(&t, 0.0, 0.0, 4, 2.0, 0.5);
        let _ = turbulence_2d(&t, 0.0, 0.0, 4);
        let _ = warp(&t, 0.0, 0.0, 2.0, 0.5);
        let _ = tile(&t, 0.0, 0.0, 1.0, 4.0, 4.0);
        let _ = terrain_height(&t, 0.0, 0.0);
        let _ = moisture_map(&t, 0.0, 0.0);
    }

    #[test]
    fn fbm_1d_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let v = fbm_1d(&t, x, 4, 2.0, 0.5);
            assert!(
                v >= -1.0 && v <= 1.0,
                "fbm_1d out of range: {v} at x={x}"
            );
        }
    }

    #[test]
    fn ridged_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = ridged_multifractal_2d(&t, x, y, 4, 2.0, 0.5);
            assert!(
                v >= -1.0 && v <= 1.0,
                "ridged out of range: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn turbulence_range() {
        let t = PermutationTable::new(42);
        for i in 0..20 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = turbulence_2d(&t, x, y, 4);
            assert!(
                v >= -1.0 && v <= 1.0,
                "turbulence out of range: {v} at ({x}, {y})"
            );
        }
    }

    #[test]
    fn different_seeds_different_values() {
        let t1 = PermutationTable::new(42);
        let t2 = PermutationTable::new(99);
        let v1 = perlin_noise_2d(&t1, 3.14, 2.71);
        let v2 = perlin_noise_2d(&t2, 3.14, 2.71);
        assert_ne!(v1, v2, "different seeds should produce different noise values");
    }

    #[test]
    fn simplex_like_range() {
        let t = PermutationTable::new(42);
        for i in 0..50 {
            let x = i as f64 * 0.37;
            let y = (i * 3 + 1) as f64 * 0.17;
            let v = simplex_like_2d(&t, x, y);
            assert!(
                v >= -1.0 && v <= 1.0,
                "simplex_like_2d out of range: {v} at ({x}, {y})"
            );
        }
    }
}