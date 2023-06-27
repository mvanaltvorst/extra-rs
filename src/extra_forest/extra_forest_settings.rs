pub enum MaxDepth {
    Infinite,
    Value(usize)
}

struct ExtraForestSettings {
    n_estimators: usize,
    max_depth: MaxDepth,
    n_jobs: usize,
}

impl Default for ExtraForestSettings {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: MaxDepth::Infinite,
            n_jobs: 1
        }
    }
}