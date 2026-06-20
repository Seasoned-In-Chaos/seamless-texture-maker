use numpy::{PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn compute_gradients(
    py: Python,
    height_map: PyReadonlyArray2<f32>,
    weights: [f32; 4],
    strength: f32,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray2<f32>>)> {
    let h_map = height_map.as_array();
    let (h, w) = h_map.dim();

    let mut normal = ndarray::Array3::<f32>::zeros((h, w, 3));
    let mut magnitude = ndarray::Array2::<f32>::zeros((h, w));

    // Compute the 9x9 kernels for X and Y
    let mut wx = [[0.0f32; 9]; 9];
    let mut wy = [[0.0f32; 9]; 9];
    let radii = [1, 2, 3, 4];
    for (i, &w_scale) in weights.iter().enumerate() {
        if w_scale == 0.0 { continue; }
        let r = radii[i];
        for dy in -r..=r {
            for dx in -r..=r {
                if dx != 0 {
                    let val = (2 * r + 1 - dx.abs() - dy.abs()) as f32;
                    wx[(dy + 4) as usize][(dx + 4) as usize] += dx.signum() as f32 * val * w_scale;
                }
                if dy != 0 {
                    let val = (2 * r + 1 - dx.abs() - dy.abs()) as f32;
                    wy[(dy + 4) as usize][(dx + 4) as usize] += dy.signum() as f32 * val * w_scale;
                }
            }
        }
    }

    // Process rows in parallel
    let rows: Vec<(usize, Vec<[f32; 3]>, f32)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row_normals = Vec::with_capacity(w);
            let mut row_mag = 0.0f32;

            for x in 0..w {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in -4..=4isize {
                    let mut cy = y as isize + ky;
                    if cy < 0 { cy += h as isize; }
                    else if cy >= h as isize { cy -= h as isize; }
                    let ucy = cy as usize;

                    for kx in -4..=4isize {
                        let w_x = wx[(ky + 4) as usize][(kx + 4) as usize];
                        let w_y = wy[(ky + 4) as usize][(kx + 4) as usize];

                        if w_x != 0.0 || w_y != 0.0 {
                            let mut cx = x as isize + kx;
                            if cx < 0 { cx += w as isize; }
                            else if cx >= w as isize { cx -= w as isize; }
                            
                            let val = h_map[[ucy, cx as usize]];
                            gx += val * w_x;
                            gy += val * w_y;
                        }
                    }
                }

                gx *= strength;
                gy *= strength;

                let nx = -gx;
                let ny = -gy;
                let nz = 1.0f32;

                let mag = (nx * nx + ny * ny + nz * nz).sqrt();

                row_normals.push([nx / mag, ny / mag, nz / mag]);

                let grad_mag = (gx * gx + gy * gy).sqrt();
                if grad_mag > row_mag {
                    row_mag = grad_mag;
                }
            }

            (y, row_normals, row_mag)
        })
        .collect();

    // Write results
    for (y, row_normals, _) in rows {
        for (x, n) in row_normals.iter().enumerate() {
            // Pack as display normal: (nz*0.5+0.5, ny*0.5+0.5, nx*0.5+0.5)
            normal[[y, x, 0]] = n[2] * 0.5 + 0.5;
            normal[[y, x, 1]] = n[1] * 0.5 + 0.5;
            normal[[y, x, 2]] = n[0] * 0.5 + 0.5;
        }
    }

    // Compute magnitude from normals
    for y in 0..h {
        for x in 0..w {
            let nx = normal[[y, x, 2]] * 2.0 - 1.0;
            let ny = normal[[y, x, 1]] * 2.0 - 1.0;
            let nz = normal[[y, x, 0]] * 2.0 - 1.0;
            magnitude[[y, x]] = (nx * nx + ny * ny).sqrt() / ((nx * nx + ny * ny + nz * nz).max(1e-8).sqrt());
        }
    }

    let normal_out = PyArray3::from_owned_array(py, normal);
    let mag_out = PyArray2::from_owned_array(py, magnitude);

    Ok((normal_out.into(), mag_out.into()))
}
