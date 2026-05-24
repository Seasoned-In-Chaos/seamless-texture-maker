use numpy::{PyArray2, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn compute_gradients(
    py: Python,
    height_map: PyReadonlyArray2<f32>,
    strength: f32,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray2<f32>>)> {
    let h_map = height_map.as_array();
    let (h, w) = h_map.dim();

    let mut normal = ndarray::Array3::<f32>::zeros((h, w, 3));
    let mut magnitude = ndarray::Array2::<f32>::zeros((h, w));

    // Sobel kernels:
    // Kx = [[-1,0,1],[-2,0,2],[-1,0,1]]
    // Ky = [[-1,-2,-1],[0,0,0],[1,2,1]]

    // Process rows in parallel
    let rows: Vec<(usize, Vec<[f32; 3]>, f32)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut row_normals = Vec::with_capacity(w);
            let mut row_mag = 0.0f32;

            for x in 0..w {
                let y0 = if y > 0 { y - 1 } else { h - 1 };
                let y2 = if y + 1 < h { y + 1 } else { 0 };
                let x0 = if x > 0 { x - 1 } else { w - 1 };
                let x2 = if x + 1 < w { x + 1 } else { 0 };

                let gx = (h_map[[y0, x2]] - h_map[[y0, x0]]
                    + 2.0 * (h_map[[y, x2]] - h_map[[y, x0]])
                    + h_map[[y2, x2]] - h_map[[y2, x0]])
                    * strength;

                let gy = (h_map[[y2, x0]] - h_map[[y0, x0]]
                    + 2.0 * (h_map[[y2, x]] - h_map[[y0, x]])
                    + h_map[[y2, x2]] - h_map[[y0, x2]])
                    * strength;

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
