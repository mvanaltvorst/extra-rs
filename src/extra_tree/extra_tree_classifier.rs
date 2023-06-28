// TODO: arbitrary number of classes.

use super::extra_tree_settings::ExtraTreeSettings;
use super::utils::{split_sample, create_subtree};
use crate::extra_tree::splitter::Splitter;
use crate::{data::tree_dataset::TreeDataset, extra_tree::node::Node};
use ndarray::{Array1, ArrayBase, Axis, Ix1, Ix2, OwnedRepr, Data};

#[derive(Debug)]
pub struct ExtraTreeClassifier {
    settings: ExtraTreeSettings,
    root: Node,
}

impl ExtraTreeClassifier {
    pub fn new(settings: ExtraTreeSettings) -> Self {
        Self {
            settings: settings,
            root: Node::Unexplored
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
            split_sample(&splitter, &dataset);

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
        self.root = create_subtree(Self::score, dataset, &self.settings);
    }

    pub fn predict_proba<T>(&self, X: &ArrayBase<T, Ix2>) -> Array1<f32> 
    where T: Data<Elem = f32> {
        X.axis_iter(Axis(0))
            .map(|x| self.root.predict(&x))
            .collect()
    }

    pub fn predict<T>(&self, X: &ArrayBase<T, Ix2>, threshold: f32) -> Array1<bool>
    where T: Data<Elem = f32> {
        // TODO: Figure out how default values in Rust arguments work.
        let probas = self.predict_proba(X);
        probas.iter().map(|v| *v > threshold).collect()
    }
}
