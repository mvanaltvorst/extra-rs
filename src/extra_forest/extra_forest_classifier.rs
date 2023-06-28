use crate::data::tree_dataset::TreeDataset;
use crate::extra_tree::extra_tree_classifier::ExtraTreeClassifier;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use crate::extra_forest::extra_forest_settings::ExtraForestSettings;
use ndarray::{ArrayBase, Ix2, OwnedRepr, Array1};

use super::extra_forest_settings::NJobs;

pub struct ExtraForestClassifier {
    tree_settings: ExtraTreeSettings,
    forest_settings: ExtraForestSettings,
    trees: Vec<ExtraTreeClassifier>,
}

impl ExtraForestClassifier {
    pub fn new(
        tree_settings: ExtraTreeSettings,
        forest_settings: ExtraForestSettings,
    ) -> ExtraForestClassifier {
        ExtraForestClassifier {
            tree_settings,
            forest_settings,
            trees: Vec::new(),
        }
    }

    pub fn fit(&mut self, dataset: &TreeDataset<bool>) {
        match self.forest_settings.n_jobs {
            NJobs::Value(0) => unreachable!("Cannot process with zero jobs."),
            NJobs::Value(1) => {
                self.trees = Vec::with_capacity(self.forest_settings.n_estimators);
                for _ in 0..self.forest_settings.n_estimators {
                    self.trees.push(self.fit_underlying_tree(dataset));
                }
            },
            NJobs::Value(_k) => {
                unimplemented!()
            },
            NJobs::NoLimit => {
                unimplemented!()
            },
        }
    }

    fn fit_underlying_tree(&self, dataset: &TreeDataset<bool>) -> ExtraTreeClassifier {
        let mut tree = ExtraTreeClassifier::new(self.tree_settings);
            
        // When bootstrapping, we use a different (random)
        // sample for each underlying tree.
        if self.forest_settings.bootstrap {
            unimplemented!();
        } else {
            tree.build(dataset);
        }

        tree
    }

    pub fn predict_proba(&self, X: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Array1<f32> {
        self.trees.iter().map(|tree| tree.predict_proba(X)).fold(
            Array1::zeros(X.len()),
            |acc, elem| acc + elem
        )
    }
}
