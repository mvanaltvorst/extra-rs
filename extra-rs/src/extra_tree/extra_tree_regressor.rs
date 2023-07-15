use crate::data::tree_dataset::TreeDataset;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use crate::extra_tree::node::Node;
use crate::extra_tree::splitter::Splitter;
use crate::extra_tree::utils::{create_subtree, split_sample};
use ndarray::{Array1, ArrayBase, Axis, Data, Ix2};

#[derive(Debug)]
pub struct ExtraTreeRegressor {
    settings: ExtraTreeSettings,
    pub root: Node,
}

impl ExtraTreeRegressor {
    pub fn new(settings: ExtraTreeSettings) -> Self {
        Self {
            settings,
            root: Node::Unexplored,
        }
    }

    fn score(splitter: &Splitter, dataset: &TreeDataset<f32>) -> f32 {
        // Decrease variance as much as possible.
        let (TreeDataset { y: ly, .. }, TreeDataset { y: ry, .. }) =
            split_sample(splitter, dataset);
        let y_mean = dataset.y.mean().unwrap();
        let ly_mean = ly.mean();
        if ly_mean.is_none() {
            return f32::NEG_INFINITY;
        }
        let ly_mean = ly_mean.unwrap();
        let ry_mean = ry.mean();
        if ry_mean.is_none() {
            return f32::NEG_INFINITY;
        }
        let ry_mean = ry_mean.unwrap();

        let y_var =
            dataset.y.iter().map(|&y| (y - y_mean).powi(2)).sum::<f32>() / dataset.y.len() as f32;
        let ly_var = ly.iter().map(|&y| (y - ly_mean).powi(2)).sum::<f32>() / ly.len() as f32;
        let ry_var = ry.iter().map(|&y| (y - ry_mean).powi(2)).sum::<f32>() / ry.len() as f32;

        let l_probability = ly.len() as f32 / dataset.y.len() as f32;
        let r_probability = ry.len() as f32 / dataset.y.len() as f32;

        (y_var - l_probability * ly_var - r_probability * ry_var) / y_var
    }

    pub fn build(&mut self, dataset: &TreeDataset<f32>) {
        self.root = create_subtree(Self::score, dataset, &self.settings, 0);
    }

    pub fn predict<T>(&self, X: &ArrayBase<T, Ix2>) -> Array1<f32>
    where
        T: Data<Elem = f32>,
    {
        X.axis_iter(Axis(0))
            .map(|x| self.root.predict(&x))
            .collect()
    }
}
