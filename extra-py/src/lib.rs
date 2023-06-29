use pyo3::prelude::*;
use ndarray::prelude::*;
// use crate::extra_tree::extra_tree_classifier::ExtraTreeClassifier;
// use crate::extra_tree::extra_tree_regressor::ExtraTreeRegressor;
// use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
// use crate::extra_forest::extra_forest_regressor::ExtraForestRegressor;
// use crate::extra_forest::extra_forest_settings::ExtraForestSettings;
// use crate::data::tree_dataset::TreeDataset;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

// #[pyfunction]
// fn tree_test_classification() -> PyResult<()> {
//     let mut tree = ExtraTreeClassifier::new(
//         ExtraTreeSettings { min_samples_split: 2, ..ExtraTreeSettings::default() }
//     );
//     println!("Tree: {:?}", tree);
//     let X = array![
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//         [1., 0., 0.],
//         [1., 1., 0.],
//     ];
//     let y = array![
//         true, false, false, true, false
//     ];
//     tree.build(&TreeDataset{ X: X.clone(), y: y.clone() });
//     println!("{:?}", tree);
//     let preds = tree.predict_proba(&X);
//     println!("{:?}", preds);
//     Ok(())
// }


// #[pyfunction]
// fn tree_test_regression() -> PyResult<()> {
//     let mut tree = ExtraTreeRegressor::new(
//         ExtraTreeSettings { min_samples_split: 2, ..ExtraTreeSettings::default() }
//     );
//     println!("Tree: {:?}", tree);
//     let X = array![
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//         [1., 0., 0.],
//         [1., 1., 0.],
//     ];
//     let y = array![
//         1.0, 0.0, 0.0, 1.0, 0.0
//     ];
//     tree.build(&TreeDataset{ X: X.clone(), y: y.clone() });
//     println!("{:#?}", tree);
//     let preds = tree.predict(&X);
//     println!("{:#?}", preds);
//     Ok(())
// }


// #[pyfunction]
// fn forest_test_regression() -> PyResult<()> {
//     let mut forest = ExtraForestRegressor::new(
//         ExtraTreeSettings::default(),
//         ExtraForestSettings { n_estimators: 500, ..ExtraForestSettings::default() },
//     );
//     println!("Forest: {:?}", forest);
//     let X = array![
//         [1., 0., 0.],
//         [0., 1., 0.],
//         [0., 0., 1.],
//         [1., 0., 0.],
//         [1., 1., 0.],
//     ];
//     let y = array![
//         1.0, 0.0, 0.0, 1.0, 0.0
//     ];
//     forest.fit(&TreeDataset{ X: X.clone(), y: y.clone() });
//     let preds = forest.predict(&X);
//     println!("{:#?}", preds);
//     Ok(())
// }

/// A Python module implemented in Rust.
#[pymodule]
fn extra_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_function(wrap_pyfunction!(tree_test_classification, m)?)?;
    // m.add_function(wrap_pyfunction!(tree_test_regression, m)?)?;
    // m.add_function(wrap_pyfunction!(forest_test_regression, m)?)?;
    Ok(())
}
