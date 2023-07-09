#[derive(Copy, Clone, Debug)]
pub enum MaxFeatures {
    Sqrt,
    Value(usize)
}

#[derive(Copy, Clone, Debug)]
pub enum MaxDepth {
    Infinite,
    Value(usize)
}

#[derive(Copy, Clone, Debug)]
pub struct ExtraTreeSettings {
    pub max_features: MaxFeatures,
    pub min_samples_split: usize,
    pub bootstrap: bool,
    pub max_depth: MaxDepth,
}

impl ExtraTreeSettings {
    pub fn new(max_features: MaxFeatures, min_samples_split: usize, bootstrap: bool, max_depth: MaxDepth) -> Self {
        Self {
            max_features,
            min_samples_split,
            bootstrap,
            max_depth,
        }
    }
}

impl Default for ExtraTreeSettings {
    fn default() -> Self {
        Self {
            max_features: MaxFeatures::Sqrt,
            min_samples_split: 2,
            bootstrap: false,
            max_depth: MaxDepth::Infinite,
        }
    }
}