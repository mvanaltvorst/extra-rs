use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, Axis};
use rand::Rng;
use crate::tree::splitter::Splitter;
use crate::tree::node::Node;
use crate::tree::tree::TreeSettings;

pub struct TreeRegressor {
    settings: TreeSettings,
    root: Box<dyn Node<f32>>,
}


impl TreeRegressor {
    pub fn new(settings: TreeSettings) -> Self {
        unimplemented!()
    }

    fn split_sample(
        splitter: &Splitter,
        X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        y: &ArrayBase<OwnedRepr<f32>, Ix1>,
    ) -> ((ArrayBase<OwnedRepr<f32>, Ix2>, ArrayBase<OwnedRepr<f32>, Ix1>), (ArrayBase<OwnedRepr<f32>, Ix2>, ArrayBase<OwnedRepr<f32>, Ix1>))
    {

        match *splitter {
            Splitter::NumericalSplitter(attribute_index, pivot) => {
                let samples = X.column(attribute_index);

                let (left_indices, right_indices) = samples.iter().enumerate().fold(
                    (Vec::new(), Vec::new()),
                    |(mut left_indices, mut right_indices), (i, &x)| {
                        if x > pivot {
                            right_indices.push(i);
                        } else {
                            left_indices.push(i);
                        }
                        (left_indices, right_indices)
                    }
                );

                let left = (
                    X.select(Axis(0), &left_indices),
                    y.select(Axis(0), &left_indices),
                );
                let right = (
                    X.select(Axis(0), &right_indices),
                    y.select(Axis(0), &right_indices),
                );

                (left, right)
            }
        }
    }

    fn score(
        splitter: &Splitter,
        X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        y: &ArrayBase<OwnedRepr<f32>, Ix1>,
    ) -> f32 {
        // Decrease variance as much as possible.
        let ((_, ly), (_, ry)) = Self::split_sample(splitter, X, y);
        let y_mean = y.mean().unwrap();
        let ly_mean = ly.mean().unwrap();
        let ry_mean = ry.mean().unwrap();

        let y_var = y.iter().map(|&y| (y - y_mean).powi(2)).sum::<f32>() / y.len() as f32;
        let ly_var = ly.iter().map(|&y| (y - ly_mean).powi(2)).sum::<f32>() / ly.len() as f32;
        let ry_var = ry.iter().map(|&y| (y - ry_mean).powi(2)).sum::<f32>() / ry.len() as f32;

        let l_probability = ly.len() as f32 / y.len() as f32;
        let r_probability = ry.len() as f32 / y.len() as f32;

        let score = (y_var - l_probability * ly_var - r_probability * ry_var) / y_var;

        score
    }
}