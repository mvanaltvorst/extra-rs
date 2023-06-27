use crate::tree::splitter::Splitter;
use crate::tree::tree::TreeSettings;
use crate::{data::tree_dataset::TreeDataset, tree::node::Node};
use ndarray::{Array1, ArrayBase, Axis, Ix1, OwnedRepr};
use super::max_features::MaxFeatures;

pub struct TreeClassifier {
    settings: TreeSettings,
    root: Node,
}

impl TreeClassifier {
    pub fn new(settings: TreeSettings) -> Self {
        unimplemented!()
    }

    fn split_sample(
        splitter: &Splitter,
        // X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        // y: &ArrayBase<OwnedRepr<bool>, Ix1>,
        dataset: &TreeDataset<bool>,
    ) -> (TreeDataset<bool>, TreeDataset<bool>) {
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

    fn score(splitter: &Splitter, dataset: &TreeDataset<bool>) -> f32 {
        // TODO: ensure this is correct. Pretty sure it's not.
        let (TreeDataset { y: ly, .. }, TreeDataset { y: ry, .. }) =
            Self::split_sample(&splitter, &dataset);

        let l_entropy = Self::labeling_entropy(&ly);
        let r_entropy = Self::labeling_entropy(&ry);

        let l_probability = ly.len() as f32 / dataset.y.len() as f32;
        let r_probability = ry.len() as f32 / dataset.y.len() as f32;

        let classification_entropy = Self::labeling_entropy(&dataset.y);

        let mean_posterior_entropy = l_probability * l_entropy + r_probability * r_entropy;

        let mutual_information = classification_entropy - mean_posterior_entropy;

        (2.0 * mutual_information) / (classification_entropy + mean_posterior_entropy)
    }

    pub fn build(
        &mut self,
        // X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        // y: &ArrayBase<OwnedRepr<bool>, Ix1>,
        dataset: &TreeDataset<bool>,
    ) {
        let is_constant: Array1<bool> = dataset
            .X
            .axis_iter(Axis(1))
            .map(|vs| vs.iter().all(|&v| v == vs[0]))
            .collect();
    }

    fn expand_node(
        node: &mut Node,
        // X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        // y: &ArrayBase<OwnedRepr<bool>, Ix1>,
        dataset: &TreeDataset<bool>,
        is_constant: &ArrayBase<OwnedRepr<bool>, Ix1>,
        settings: &TreeSettings,
    ) {
        if TreeClassifier::stop_expasion(dataset, is_constant, settings) {
            // we only support two classes: 0 and 1.
            // our prediction is the percentage of 1's occuring.
            let prediction: f32 = dataset
                .y
                .iter()
                .map(|v| if *v { 1.0 } else { 0.0 })
                .sum::<f32>()
                / (dataset.y.len() as f32);
            *node = Node::Leaf(prediction);
        } else {
            // In case of automatic max features in a classification problem,
            // we take the length of the dataset as number of splits to consider.
            // TODO: reference
            let k = match settings.max_features {
                MaxFeatures::Auto => dataset.y.len(),
                MaxFeatures::Value(k) => k,
            };

            // take k random features without replacement.
            let rand_indices = {
                let mut indices: Vec<usize> = (0..is_constant.len()).filter(
                    |&i| !is_constant[i],
                ).collect();

                let mut rng = rand::thread_rng();
                indices.shuffle(&mut rng);
                indices.iter().take(k.min(indices.len())).cloned().collect()
            };


        }
    }

    fn stop_expasion(
        // X: &ArrayBase<OwnedRepr<f32>, Ix2>,
        // y: &ArrayBase<OwnedRepr<bool>, Ix1>,
        dataset: &TreeDataset<bool>,
        is_constant: &ArrayBase<OwnedRepr<bool>, Ix1>,
        settings: &TreeSettings,
    ) -> bool {
        // should be at least `min_samples_split` in the dataset.
        if dataset.y.len() < settings.min_samples_split {
            return true;
        }

        // if no variables anymore
        if is_constant.iter().all(|&x| x) {
            return true;
        }

        // if y is constant
        if dataset.y.iter().all(|&v| v == dataset.y[0]) {
            return true;
        }

        false
    }
}
