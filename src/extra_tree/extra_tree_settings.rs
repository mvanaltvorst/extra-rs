pub enum MaxFeatures {
    Sqrt,
    Value(usize)
}

pub struct ExtraTreeSettings {
    pub max_features: MaxFeatures,
    pub min_samples_split: usize,
    pub bootstrap: bool,
}

impl ExtraTreeSettings {
    pub fn new(max_features: MaxFeatures, min_samples_split: usize, bootstrap: bool) -> Self {
        assert!(!bootstrap, "bootstrap not implemented");

        Self {
            max_features: max_features,
            min_samples_split: min_samples_split,
            bootstrap: bootstrap
        }
    }
}

impl Default for ExtraTreeSettings {
    fn default() -> Self {
        Self {
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 20,
            bootstrap: false
        }
    }
}