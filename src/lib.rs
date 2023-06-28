use pyo3::prelude::*;
use ndarray::prelude::*;
use crate::extra_tree::extra_tree_classifier::ExtraTreeClassifier;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use crate::data::tree_dataset::TreeDataset;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn tree_test() -> PyResult<()> {
    println!("Test!");
    let mut tree = ExtraTreeClassifier::new(
        ExtraTreeSettings::default()
    );
    println!("Tree: {:?}", tree);
    let X = array![
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [1., 0., 0.],
        [1., 1., 0.],
    ];
    let y = array![
        true, false, false, true, false
    ];
    tree.build(&TreeDataset{ X, y });
    println!("{:?}", tree);
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn extra_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(tree_test, m)?)?;
    Ok(())
}


pub mod data;
pub mod extra_forest;
pub mod extra_tree;