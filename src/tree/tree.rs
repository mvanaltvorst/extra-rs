use ndarray::{ArrayBase, Ix1, Data};
use rand::Rng;
use crate::tree::splitter::{Splitter, NumericalSplitter};
use crate::tree::node::Node;

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

pub trait Tree {
    fn pick_random_split<T>(samples: ArrayBase<T, Ix1>, attribute_index: usize) -> Box<dyn Splitter>
    where T: Data<Elem = f32>;
}

pub struct TreeClassifier {
    root: dyn Node<bool>
}


pub struct TreeRegressor {
    root: dyn Node<f32>
}

impl Tree for TreeClassifier {
    fn pick_random_split<T>(samples: ArrayBase<T, Ix1>, attribute_index: usize) -> Box<dyn Splitter> 
    where T: Data<Elem = f32> {
        let (min, max) = samples.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
            (min.min(x), max.max(x))
        });
        let pivot = rand::thread_rng().gen::<f32>() * (max - min) + min;
        Box::new(NumericalSplitter::new(attribute_index, pivot))
    }
}

