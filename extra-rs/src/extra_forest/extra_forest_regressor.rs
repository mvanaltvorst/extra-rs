use crate::data::tree_dataset::TreeDataset;
use crate::extra_tree::extra_tree_regressor::ExtraTreeRegressor;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use crate::extra_forest::extra_forest_settings::ExtraForestSettings;
use ndarray::{ArrayBase, Ix2, OwnedRepr, Array1};
use rayon::prelude::*;

use super::extra_forest_settings::NJobs;

#[derive(Debug)]
pub struct ExtraForestRegressor {
    pub tree_settings: ExtraTreeSettings,
    pub forest_settings: ExtraForestSettings,
    pub trees: Vec<ExtraTreeRegressor>,
}

impl ExtraForestRegressor {
    pub fn new(
        tree_settings: ExtraTreeSettings,
        forest_settings: ExtraForestSettings,
    ) -> ExtraForestRegressor {
        ExtraForestRegressor {
            tree_settings,
            forest_settings,
            trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, dataset: &TreeDataset<f32>) {
        match self.forest_settings.n_jobs {
            NJobs::Value(0) => panic!("Cannot process with zero jobs."),
            NJobs::Value(1) => {
                self.trees = Vec::with_capacity(self.forest_settings.n_estimators);
                for _ in 0..self.forest_settings.n_estimators {
                    self.trees.push(self.fit_underlying_tree(dataset));
                }
            },
            NJobs::Value(_k) => {
                // we use rayon with k cores
                unimplemented!("Use the `RAYON_NUM_THREADS` environment variable to set the number of threads.");
            },
            NJobs::NoLimit => {
                // we use rayon
                self.trees = (0..self.forest_settings.n_estimators).into_par_iter().map(|_| {
                    self.fit_underlying_tree(dataset)
                }).collect();
            },
        }
    }

    fn fit_underlying_tree(&self, dataset: &TreeDataset<f32>) -> ExtraTreeRegressor {
        let mut tree = ExtraTreeRegressor::new(self.tree_settings);
            
        // When bootstrapping, we use a different (random)
        // sample for each underlying tree.
        if self.forest_settings.bootstrap {
            unimplemented!();
        } else {
            tree.build(dataset);
        }

        tree
    }

    pub fn predict(&self, X: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Array1<f32> {
        let n_obs = X.shape()[0];
        match self.forest_settings.n_jobs {
            NJobs::Value(0) => panic!("Cannot process with zero jobs."),
            NJobs::Value(1) => {
                self.trees.iter().map(|tree| tree.predict(X)).reduce(
                    |acc, elem| acc + elem
                ).unwrap() / (self.forest_settings.n_estimators as f32)
            },
            NJobs::Value(_k) => {
                // we use rayon with k cores
                unimplemented!("Use the `RAYON_NUM_THREADS` environment variable to set the number of threads.");
            },
            NJobs::NoLimit => {
                // we use rayon
                self.trees.par_iter().map(|tree| tree.predict(X)).reduce(
                    || Array1::zeros(n_obs),
                    |acc, elem| acc + elem
                ) / (self.forest_settings.n_estimators as f32)
            },
        }
    }
}
