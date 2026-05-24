use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

#[pyfunction]
pub fn splat_synthesize(
    py: Python,
    base: PyReadonlyArray3<f32>,
    patch_size: usize,
    density: f32,
    seed: u64,
    wrap_edges: bool,
) -> PyResult<Py<PyArray3<f32>>> {
    let base_arr = base.as_array();
    let (h, w, _c) = base_arr.dim();

    let cell_size = (patch_size as f32 * 0.5).max(4.0);
    let cells_x = ((w as f32 / cell_size) * density).ceil() as usize + 2;
    let cells_y = ((h as f32 / cell_size) * density).ceil() as usize + 2;
    let num_cells = cells_x * cells_y;

    let half_patch = patch_size / 2;

    let sigma = patch_size as f32 / 4.0;
    let sigma2 = 2.0 * sigma * sigma;
    let alpha_mask: Vec<f32> = (0..patch_size)
        .flat_map(|dy| {
            (0..patch_size).map(move |dx| {
                let dx2 = (dx as f32 - half_patch as f32).powi(2);
                let dy2 = (dy as f32 - half_patch as f32).powi(2);
                (-(dx2 + dy2) / sigma2).exp()
            })
        })
        .collect();

    let cell_data: Vec<(usize, usize, usize, Pcg64)> = (0..num_cells)
        .map(|idx| {
            let gy = idx / cells_x;
            let gx = idx % cells_x;
            let cell_seed = seed ^ (idx as u64).wrapping_mul(0x517cc1b727220a95);
            let rng = Pcg64::seed_from_u64(cell_seed);
            (gx, gy, idx, rng)
        })
        .collect();

    let mut output = ndarray::Array3::<f32>::zeros((h, w, 3));

    for y in 0..h {
        for x in 0..w {
            let by = y % h;
            let bx = x % w;
            for ch in 0..3 {
                output[[y, x, ch]] = base_arr[[by, bx, ch]];
            }
        }
    }

    for (gx, gy, _idx, mut rng) in cell_data {
        let cx = (gx as f32 + 0.5) * cell_size - cell_size;
        let cy = (gy as f32 + 0.5) * cell_size - cell_size;

        let wobble_x = (rng.gen::<f32>() - 0.5) * cell_size * 0.4;
        let wobble_y = (rng.gen::<f32>() - 0.5) * cell_size * 0.4;

        let splat_cx = cx + wobble_x;
        let splat_cy = cy + wobble_y;

        let px = rng.gen_range(0..w.saturating_sub(patch_size).max(1));
        let py = rng.gen_range(0..h.saturating_sub(patch_size).max(1));

        for dy in 0..patch_size {
            for dx in 0..patch_size {
                let alpha = alpha_mask[dy * patch_size + dx];

                if alpha < 0.001 {
                    continue;
                }

                let out_x = (splat_cx as isize + dx as isize - half_patch as isize) as usize;
                let out_y = (splat_cy as isize + dy as isize - half_patch as isize) as usize;

                if wrap_edges {
                    let ox = out_x % w;
                    let oy = out_y % h;
                    let inv = 1.0 - alpha;
                    for ch in 0..3 {
                        output[[oy, ox, ch]] =
                            inv * output[[oy, ox, ch]] + alpha * base_arr[[(py + dy) % h, (px + dx) % w, ch]];
                    }
                } else {
                    if out_x < w && out_y < h {
                        let inv = 1.0 - alpha;
                        for ch in 0..3 {
                            output[[out_y, out_x, ch]] =
                                inv * output[[out_y, out_x, ch]]
                                    + alpha * base_arr[[(py + dy) % h, (px + dx) % w, ch]];
                        }
                    }
                }
            }
        }
    }

    let out = PyArray3::from_owned_array(py, output);
    Ok(out.into())
}
