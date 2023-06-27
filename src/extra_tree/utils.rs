use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, Axis};
use rand::Rng;
use crate::extra_tree::splitter::Splitter;
use crate::extra_tree::{extra_tree_classifier::TreeClassifier, extra_tree_regressor::TreeRegressor};
use crate::extra_tree::max_features::MaxFeatures;


pub fn pick_random_split(samples: ArrayBase<OwnedRepr<f32>, Ix1>, attribute_index: usize) -> Splitter {
    let (min, max) = samples.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
        (min.min(x), max.max(x))
    });
    let pivot = rand::thread_rng().gen::<f32>() * (max - min) + min;
    Splitter::NumericalSplitter(attribute_index, pivot)
}
