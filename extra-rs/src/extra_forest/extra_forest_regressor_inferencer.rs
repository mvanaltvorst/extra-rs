use super::extra_forest_regressor::ExtraForestRegressor;
use super::extra_forest_settings::NJobs;
use crate::extra_tree::extra_tree_regressor_inferencer::{ExtraTreeRegressorInferencer, fast_sigmoid, inverse_fast_sigmoid};

pub struct ExtraForestRegressorInferencer {
    // We can turn an ExtraForestRegressor into an ExtraForestRegressorInferencer
    pub trees: Vec<ExtraTreeRegressorInferencer>,
}

impl ExtraForestRegressorInferencer {
    pub fn new(
        extra_forest: &ExtraForestRegressor,
    ) -> Result<ExtraForestRegressorInferencer, String> {
        let mut trees = Vec::with_capacity(extra_forest.trees.len());
        for tree in &extra_forest.trees {
            trees.push(ExtraTreeRegressorInferencer::new(&tree)?);
        }
        Ok(ExtraForestRegressorInferencer { trees })
    }

    pub fn predict(&self, x: &[f32]) -> f32 {
        // Multithreading does not make sense here given the overhead of spawning threads.
        let quantized = x.iter().map(|x| fast_sigmoid(*x)).collect::<Vec<u8>>();
        self.trees.iter().map(|tree| inverse_fast_sigmoid(tree.predict_quantized(&quantized))).sum::<f32>() / (self.trees.len() as f32)
    }
}
