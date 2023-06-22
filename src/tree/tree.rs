use crate::data::{tree_dataset::TreeDataset, tree_row::TreeRow};

pub struct TreeSettings {
    pub max_features: usize,
    pub min_samples_split: usize,
}

impl TreeSettings {
    pub fn new(max_features: usize, min_samples_split: usize, n_classes: usize, n_outputs: usize) -> Self {
        Self {
            max_features,
            min_samples_split,
        }
    }
}

pub trait Tree {
    pub fn new(settings: TreeSettings);
    fn pick_random_split(samples: Vec<TreeRow>, attribute_inde
}

pub struct TreeClassifier {
    pub tree: Tree
}

pub struct TreeRegressor {
    pub tree: Tree
}
