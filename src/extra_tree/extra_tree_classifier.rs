use super::extra_tree_settings::{ExtraTreeSettings, MaxFeatures};
use super::utils::pick_random_split;
use crate::extra_tree::splitter::Splitter;
use crate::{data::tree_dataset::TreeDataset, extra_tree::node::Node};
use ndarray::{Array1, ArrayBase, Axis, Ix1, Ix2, OwnedRepr, Data};
use rand::seq::SliceRandom;

pub struct ExtraTreeClassifier {
    settings: ExtraTreeSettings,
    root: Node,
}

impl ExtraTreeClassifier {
    pub fn new(settings: ExtraTreeSettings) -> Self {
        unimplemented!()
    }

    fn split_sample(
        splitter: &Splitter,
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
        dataset: &TreeDataset<bool>,
    ) {
        self.root = Self::create_subtree(dataset, &self.settings);
    }

    fn create_subtree(dataset: &TreeDataset<bool>, settings: &ExtraTreeSettings) -> Node {
        let is_constant: Array1<bool> = dataset
            .X
            .axis_iter(Axis(1))
            .map(|vs| vs.iter().all(|&v| v == vs[0]))
            .collect();
        if Self::stop_expasion(dataset, &is_constant, settings) {
            // we only support two classes: 0 and 1.
            // our prediction is the percentage of 1's occuring.
            let prediction: f32 = dataset
                .y
                .iter()
                .map(|v| if *v { 1.0 } else { 0.0 })
                .sum::<f32>()
                / (dataset.y.len() as f32);
            Node::Leaf(prediction)
        } else {
            // In case of automatic max features in a classification problem,
            // we take the length of the dataset as number of splits to consider.
            // TODO: reference
            let k = match settings.max_features {
                MaxFeatures::Sqrt => dataset.y.len(),
                MaxFeatures::Value(k) => k,
            };

            // take k random features without replacement.
            let rand_indices = {
                let mut indices: Vec<usize> = (0..is_constant.len())
                    .filter(|&i| !is_constant[i])
                    .collect();

                let mut rng = rand::thread_rng();
                // TODO: figure out how to do this without
                // shuffling the entire list.
                indices.shuffle(&mut rng);
                indices.iter().take(k.min(indices.len())).cloned().collect::<Vec<usize>>()
            };

            let X_feature_subset = dataset.X.select(Axis(1), &rand_indices);

            let best_split = (0..rand_indices.len())
                .map(|i| pick_random_split(X_feature_subset.index_axis(Axis(1), i).to_owned(), rand_indices[i]))
                .map(|splitter| (Self::score(&splitter, &dataset), splitter))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                .unwrap()
                .1;

            let (left, right) = Self::split_sample(&best_split, &dataset);

            Node::Branch(
                best_split,
                Box::new(Self::create_subtree(&left, settings)),
                Box::new(Self::create_subtree(&right, settings)),
            )
        }
    }

    fn stop_expasion(
        dataset: &TreeDataset<bool>,
        is_constant: &ArrayBase<OwnedRepr<bool>, Ix1>,
        settings: &ExtraTreeSettings,
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

    fn predict_proba(&self, X: &ArrayBase<OwnedRepr<f32>, Ix2>) -> Array1<f32> {
        X.axis_iter(Axis(0))
            .map(|x| Self::predict_proba_node_walk(&self.root, &x))
            .collect()
    }

    fn predict_proba_node_walk<T>(node: &Node, x: &ArrayBase<T, Ix1>) -> f32
    where T: Data<Elem = f32> {
        match node {
            Node::Leaf(p) => *p,
            Node::Branch(splitter, left, right) => {
                if splitter.split(x) {
                    Self::predict_proba_node_walk(right, x)
                } else {
                    Self::predict_proba_node_walk(left, x)
                }
            }
        }
    }
}
