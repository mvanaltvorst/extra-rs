use crate::tree::node::Node;
use crate::tree::splitter::Splitter;
use crate::tree::tree::TreeSettings;
use ndarray::{ArrayBase, Axis, Ix1, Ix2, OwnedRepr};
use rand::Rng;

pub struct TreeClassifier {
    settings: TreeSettings,
    root: Box<dyn Node<bool>>,
}

impl TreeClassifier {
    pub fn new(settings: TreeSettings) -> Self {
        unimplemented!()
    }

    fn split_sample(
        splitter: &Splitter,
        X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        y: &ArrayBase<OwnedRepr<bool>, Ix1>,
    ) -> (
        (
            ArrayBase<OwnedRepr<f32>, Ix2>,
            ArrayBase<OwnedRepr<bool>, Ix1>,
        ),
        (
            ArrayBase<OwnedRepr<f32>, Ix2>,
            ArrayBase<OwnedRepr<bool>, Ix1>,
        ),
    ) {
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
                    },
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

    fn labeling_entropy(y: &ArrayBase<OwnedRepr<bool>, Ix1>) -> f32 {
        let mut p = 0.0;
        let mut n = 0.0;
        for &y in y.iter() {
            if y {
                p += 1.0;
            } else {
                n += 1.0;
            }
        }
        let p = p / y.len() as f32;
        let n = n / y.len() as f32;
        if p == 0.0 || n == 0.0 {
            0.0
        } else {
            -p * p.log2() - n * n.log2()
        }
    }

    fn score(
        splitter: Splitter,
        X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        y: &ArrayBase<OwnedRepr<bool>, Ix1>,
    ) -> f32 {
        // TODO: ensure this is correct. Pretty sure it's not.
        let ((_, ly), (_, ry)) = Self::split_sample(&splitter, X, y);

        let l_entropy = Self::labeling_entropy(&ly);
        let r_entropy = Self::labeling_entropy(&ry);

        let l_probability = ly.len() as f32 / y.len() as f32;
        let r_probability = ry.len() as f32 / y.len() as f32;

        let classification_entropy = Self::labeling_entropy(y);

        let mean_posterior_entropy = l_probability * l_entropy + r_probability * r_entropy;

        let mutual_information = classification_entropy - mean_posterior_entropy;

        let score = (2.0 * mutual_information) / (classification_entropy + mean_posterior_entropy);
        score
    }
}
