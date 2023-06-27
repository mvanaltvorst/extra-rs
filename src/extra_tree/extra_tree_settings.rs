pub enum MaxFeatures {
    Auto,
    Value(usize)
}

pub struct ExtraTreeSettings {
    pub max_features: MaxFeatures,
    pub min_samples_split: usize,
    pu
}

impl ExtraTreeSettings {
    pub fn new(max_features: MaxFeatures, min_samples_split: usize) -> Self {
        Self {
            max_features,
            min_samples_split,
        }
    }
}