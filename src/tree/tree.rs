use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, Axis};
use rand::Rng;
use crate::tree::splitter::Splitter;
use crate::tree::{tree_classifier::TreeClassifier, tree_regressor::TreeRegressor};

pub struct TreeSettings {
    pub max_features: usize,
    pub min_samples_split: usize,
}

impl TreeSettings {
    pub fn new(max_features: usize, min_samples_split: usize) -> Self {
        Self {
            max_features,
            min_samples_split,
        }
    }
}

pub enum Tree {
    // A tree contains settings and a root
    TreeClassifier(TreeClassifier), 
    TreeRegressor(TreeRegressor),
}


impl Tree {
    fn pick_random_split(samples: ArrayBase<OwnedRepr<f32>, Ix1>, attribute_index: usize) -> Splitter {
        let (min, max) = samples.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
        let pivot = rand::thread_rng().gen::<f32>() * (max - min) + min;
        Splitter::NumericalSplitter(attribute_index, pivot)
    }
}
