#[derive(Copy, Clone, Debug)]
pub enum MaxFeatures {
    Sqrt,
    Value(usize)
}

#[derive(Copy, Clone, Debug)]
pub struct ExtraTreeSettings {
    pub max_features: MaxFeatures,
    pub min_samples_split: usize,
}

impl ExtraTreeSettings {
    pub fn new(max_features: MaxFeatures, min_samples_split: usize, _bootstrap: bool) -> Self {
        Self {
            max_features,
            min_samples_split,
        }
    }
}

impl Default for ExtraTreeSettings {
    fn default() -> Self {
        Self {
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
        }
    }
}