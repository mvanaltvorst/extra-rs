use crate::data::tree_dataset::TreeDataset;
use crate::extra_tree::node::Node;
use crate::extra_tree::splitter::Splitter;
use crate::extra_tree::utils::{split_sample, create_subtree};
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use ndarray::{ArrayBase, Axis, Ix2, Array1, Data};

pub struct TreeRegressor {
    settings: ExtraTreeSettings,
    root: Node,
}

impl TreeRegressor {
    pub fn new(settings: ExtraTreeSettings) -> Self {
        Self {
            settings: settings,
            root: Node::Unexplored,
        }
    }

    fn score(
        splitter: &Splitter,
        dataset: &TreeDataset<f32>
    ) -> f32 {
        // Decrease variance as much as possible.
        let (TreeDataset{y: ly, ..}, TreeDataset{y: ry, ..}) = split_sample(splitter, &dataset);
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

    pub fn build(
        &mut self,
        dataset: &TreeDataset<f32>
    ) {
        self.root = create_subtree(Self::score, dataset, &self.settings);
    }

    pub fn predict<T>(&self, X: &ArrayBase<T, Ix2>) -> Array1<f32> 
    where T: Data<Elem = f32> {
        X.axis_iter(Axis(0))
            .map(|x| self.root.predict(&x))
            .collect()
    }
}
