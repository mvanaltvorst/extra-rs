use crate::extra_tree::extra_tree_classifier::ExtraTreeClassifier;
use crate::extra_tree::extra_tree_settings::ExtraTreeSettings;

struct ExtraForestClassifier {
    tree_settings: ExtraTreeSettings,
    forest_settings: ExtraForestSettings
    trees: Vec<ExtraTreeClassifier>
}

impl ExtraForestClassifier {
    fn new(settings: ExtraTreeSettings) -> ExtraForestClassifier {
        ExtraForestClassifier {
            trees: Vec::new()
        }
    }
}