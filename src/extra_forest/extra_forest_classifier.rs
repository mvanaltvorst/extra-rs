use crate::extra_tree::extra_tree_classifier::ExtraTreeClassifier;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;
use crate::data::tree_dataset::TreeDataset;

struct ExtraForestClassifier {
    tree_settings: ExtraTreeSettings,
    forest_settings: ExtraForestSettings,
    trees: Vec<ExtraTreeClassifier>,
}

impl ExtraForestClassifier {
    fn new(
        tree_settings: ExtraTreeSettings,
        forest_settings: ExtraForestSettings,
    ) -> ExtraForestClassifier {
        ExtraForestClassifier {
            tree_settings: tree_settings,
            forest_settings: forest_settings,
            trees: Vec::with_capacity(forest_settings.n_trees),
        }
    }

    fn fit(&mut self, dataset: &TreeDataset<bool>) {
        for 
    }
}
