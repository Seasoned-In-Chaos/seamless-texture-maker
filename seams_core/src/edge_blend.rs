use numpy::{PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

#[pyfunction]
pub fn edge_blend_symmetric(
    py: Python,
    img: PyReadonlyArray3<f32>,
    blend_width: usize,
    symmetric: bool,
) -> PyResult<Py<PyArray3<f32>>> {
    let img_arr = img.as_array();
    let (h, w, c) = img_arr.dim();
    let half = blend_width / 2;

    let mut result = img_arr.to_owned();

    let weights: Vec<f32> = (1..=half)
        .map(|i| {
            let t = i as f32 / half as f32;
            0.25 * (std::f32::consts::PI * t).cos()
        })
        .collect();

    let cx = w / 2;
    let cy = h / 2;

    // Horizontal seam: blend columns around cx
    for y in 0..h {
        for (i, &weight) in weights.iter().enumerate() {
            let offset = i + 1;
            let inv = 1.0 - weight;

            let left_col = if cx >= offset { cx - offset } else { w - (offset - cx) };
            let right_col = (cx + offset) % w;

            if symmetric {
                let mut tmp = vec![0.0f32; c];
                for ch in 0..c {
                    tmp[ch] = result[[y, left_col, ch]];
                }
                for ch in 0..c {
                    let left_val = tmp[ch];
                    let right_val = result[[y, right_col, ch]];
                    result[[y, left_col, ch]] = inv * left_val + weight * right_val;
                    result[[y, right_col, ch]] = inv * right_val + weight * left_val;
                }
            } else {
                for ch in 0..c {
                    let left_val = result[[y, left_col, ch]];
                    let right_val = result[[y, right_col, ch]];
                    result[[y, left_col, ch]] = inv * left_val + weight * right_val;
                }
            }
        }
    }

    // Vertical seam: blend rows around cy
    for x in 0..w {
        for (i, &weight) in weights.iter().enumerate() {
            let offset = i + 1;
            let inv = 1.0 - weight;

            let top_row = if cy >= offset { cy - offset } else { h - (offset - cy) };
            let bottom_row = (cy + offset) % h;

            if symmetric {
                let mut tmp = vec![0.0f32; c];
                for ch in 0..c {
                    tmp[ch] = result[[top_row, x, ch]];
                }
                for ch in 0..c {
                    let top_val = tmp[ch];
                    let bottom_val = result[[bottom_row, x, ch]];
                    result[[top_row, x, ch]] = inv * top_val + weight * bottom_val;
                    result[[bottom_row, x, ch]] = inv * bottom_val + weight * top_val;
                }
            } else {
                for ch in 0..c {
                    let top_val = result[[top_row, x, ch]];
                    let bottom_val = result[[bottom_row, x, ch]];
                    result[[top_row, x, ch]] = inv * top_val + weight * bottom_val;
                }
            }
        }
    }

    let out = PyArray3::from_owned_array(py, result);
    Ok(out.into())
}
