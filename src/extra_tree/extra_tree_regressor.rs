use crate::data::{tree_dataset::TreeDataset, self};
use crate::extra_tree::node::Node;
use crate::extra_tree::splitter::Splitter;
use crate::extra_tree::utils::ExtraTreeSettings;
use ndarray::{ArrayBase, Axis, Ix1, Ix2, OwnedRepr};
use rand::Rng;

use super::utils::Tree;

pub struct TreeRegressor {
    settings: ExtraTreeSettings,
    root: Node,
}

impl TreeRegressor {
    pub fn new(settings: ExtraTreeSettings) -> Self {
        unimplemented!()
    }

    fn split_sample(
        splitter: &Splitter,
        dataset: &TreeDataset<f32>,
    ) -> (
        TreeDataset<f32>,
        TreeDataset<f32>,
    ) {
        match *splitter {
            Splitter::NumericalSplitter(attribute_index, pivot) => {
                let samples = dataset.X.column(attribute_index);

                let (left_indices, right_indices) = samples.iter().enumerate().fold(
                    (Vec::new(), Vec::new()),
                    |(mut left_indices, mut right_indices), (i, &x)| {
                        if x > pivot {
                            right_indices.push(i);
                        } else {
                            left_indices.push(i);
                        }
                        (left_indices, right_indices)
                    },
                );

                let left = TreeDataset { 
                    X: dataset.X.select(Axis(0), &left_indices),
                    y: dataset.y.select(Axis(0), &left_indices),
                };
                let right = TreeDataset { 
                    X: dataset.X.select(Axis(0), &right_indices),
                    y: dataset.y.select(Axis(0), &right_indices),
                };

                (left, right)
            }
        }
    }

    fn score(
        splitter: &Splitter,
        dataset: &TreeDataset<f32>
    ) -> f32 {
        // Decrease variance as much as possible.
        let (TreeDataset{y: ly, ..}, TreeDataset{y: ry, ..}) = Self::split_sample(splitter, &dataset);
        let y_mean = dataset.y.mean().unwrap();
        let ly_mean = ly.mean().unwrap();
        let ry_mean = ry.mean().unwrap();

        let y_var = dataset.y.iter().map(|&y| (y - y_mean).powi(2)).sum::<f32>() / dataset.y.len() as f32;
        let ly_var = ly.iter().map(|&y| (y - ly_mean).powi(2)).sum::<f32>() / ly.len() as f32;
        let ry_var = ry.iter().map(|&y| (y - ry_mean).powi(2)).sum::<f32>() / ry.len() as f32;

        let l_probability = ly.len() as f32 / dataset.y.len() as f32;
        let r_probability = ry.len() as f32 / dataset.y.len() as f32;

        let score = (y_var - l_probability * ly_var - r_probability * ry_var) / y_var;

        score
    }
}
