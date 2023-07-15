use extra_rs::extra_forest::extra_forest_regressor::ExtraForestRegressor;
use extra_rs::extra_forest::extra_forest_settings::{ExtraForestSettings, NJobs};
use extra_rs::extra_tree::extra_tree_settings::{ExtraTreeSettings, MaxDepth, MaxFeatures};
use extra_rs::data::tree_dataset::TreeDataset;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2, PyArray1};
// use numpy::ndarray::P;


#[pyclass(name = "ExtraForestRegressor")]
pub struct PyExtraForestRegressor {
    inner: ExtraForestRegressor,
}

#[pymethods]
impl PyExtraForestRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=100, max_depth=-1, n_jobs=-1, bootstrap=false, max_features="sqrt", min_samples_split=2))]
    fn new(
        n_estimators: usize,
        max_depth: i32,
        n_jobs: i32,
        bootstrap: bool,
        max_features: &str,
        min_samples_split: usize,
    ) -> PyResult<Self> {
        let forest_settings = {
            let mut forest_settings = ExtraForestSettings::default();
            forest_settings.n_estimators = n_estimators;
            if n_jobs <= -2 || n_jobs == 0 {
                return Err(PyValueError::new_err("`n_jobs` must be greater than 0 or must be -1."));
            } else if n_jobs == -1 {
                // we parse -1 as no limit
                forest_settings.n_jobs = NJobs::NoLimit;
            } else {
                forest_settings.n_jobs = NJobs::Value(n_jobs as usize);
            }
            forest_settings.bootstrap = bootstrap;
            forest_settings
        };

        let tree_settings = {
            let mut tree_settings = ExtraTreeSettings::default();

            if max_depth <= -2 {
                return Err(PyValueError::new_err("max_depth must be greater than -2"));
            } else if max_depth == -1 {
                // we parse -1 as infinite depth
                tree_settings.max_depth = MaxDepth::Infinite;
            } else {
                tree_settings.max_depth = MaxDepth::Value(max_depth as usize);
            }

            if max_features == "sqrt" {
                // we parse 1 as no limit
                tree_settings.max_features = MaxFeatures::Sqrt;
            } else {
                // try to parse as usize
                if let Ok(max_features) = max_features.parse::<usize>() {
                    tree_settings.max_features = MaxFeatures::Value(max_features);
                } else {
                    // not valid, raise error
                    return Err(PyValueError::new_err(
                        "max_features must be 'sqrt' or a positive integer",
                    ));
                }
            }
            if min_samples_split <= 0 {
                return Err(PyValueError::new_err(
                    "min_samples_split must be greater than 0",
                ));
            } else {
                tree_settings.min_samples_split = min_samples_split;
            }
            tree_settings
        };

        Ok(PyExtraForestRegressor {
            inner: ExtraForestRegressor::new(tree_settings, forest_settings),
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f32>, y: PyReadonlyArray1<f32>) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        let dataset: TreeDataset<f32> = TreeDataset {
            X: x.to_owned(),
            y: y.to_owned(),
        };
        self.inner.fit(&dataset);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<f32>) -> &'py PyArray1<f32> {
        let x = x.as_array();
        self.inner.predict(&x.to_owned()).into_pyarray(py)
    }

    fn get_debug_tree_descriptions(&self) -> Vec<String> {
        self.inner.trees.iter().map(|tree| format!("{:?}", tree.root)).collect()
    }
}
