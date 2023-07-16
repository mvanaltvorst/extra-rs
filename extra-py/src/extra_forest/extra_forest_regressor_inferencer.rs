use extra_rs::data::tree_dataset::TreeDataset;
use extra_rs::extra_forest::extra_forest_regressor::ExtraForestRegressor;
use extra_rs::extra_forest::extra_forest_regressor_inferencer::ExtraForestRegressorInferencer;
use extra_rs::extra_forest::extra_forest_settings::{ExtraForestSettings, NJobs};
use extra_rs::extra_tree::extra_tree_settings::{ExtraTreeSettings, MaxDepth, MaxFeatures};
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass(name = "ExtraForestRegressorInferencer")]
pub struct PyExtraForestRegressorInferencer {
    inner: ExtraForestRegressorInferencer,
    n_jobs: NJobs
}

impl PyExtraForestRegressorInferencer {
    pub fn new(extra_forest_regressor: &ExtraForestRegressor) -> PyResult<Self> {
        Ok(PyExtraForestRegressorInferencer {
            inner: ExtraForestRegressorInferencer::new(extra_forest_regressor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            n_jobs: extra_forest_regressor.forest_settings.n_jobs
        })
    }
}

#[pymethods]
impl PyExtraForestRegressorInferencer {
    pub fn predict<'py>(&self, py: Python<'py>, X: PyReadonlyArray2<f32>) -> &'py PyArray1<f32> {
        let X = X.as_array();
        // we iterate over rows
        let num_rows = X.shape()[0];
        // let mut y = Vec::with_capacity(num_rows);
        // for i in 0..num_rows {
        //     y.push(
        //         self.inner
        //             .predict(&X.slice(s![i, ..]).iter().map(|x| *x).collect::<Vec<f32>>()),
        //     );
        // }
        let y = match self.n_jobs {
            NJobs::NoLimit => {
                (0..num_rows).into_par_iter().map(|i| {
                    self.inner
                        .predict(&X.slice(s![i, ..]).iter().map(|x| *x).collect::<Vec<f32>>())
                }).collect::<Vec<f32>>()
            },
            NJobs::Value(1) => {
                (0..num_rows).into_iter().map(|i| {
                    self.inner
                        .predict(&X.slice(s![i, ..]).iter().map(|x| *x).collect::<Vec<f32>>())
                }).collect::<Vec<f32>>()
            },
            NJobs::Value(0) => panic!("n_jobs cannot be 0"),
            NJobs::Value(k) => unimplemented!("Set RAYON_NUM_THREADS=k to use k threads"),
        };
        y.to_pyarray(py)
    }

    pub fn get_debug_tree_descriptions(&self) -> Vec<String> {
        self.inner.trees.iter().map(|tree|tree.get_debug_tree_description()).collect()
    }
}
