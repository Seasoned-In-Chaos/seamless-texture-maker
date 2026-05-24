use pyo3::prelude::*;

mod edge_blend;
mod splat;
mod gradients;

#[pymodule]
fn seams_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(edge_blend::edge_blend_symmetric, m)?)?;
    m.add_function(wrap_pyfunction!(splat::splat_synthesize, m)?)?;
    m.add_function(wrap_pyfunction!(gradients::compute_gradients, m)?)?;
    Ok(())
}
