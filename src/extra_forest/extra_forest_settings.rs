use std::marker::Copy;

#[derive(Copy, Clone)]
pub enum MaxDepth {
    Infinite,
    Value(usize)
}

#[derive(Copy, Clone)]
pub enum NJobs {
    NoLimit,
    Value(usize)
}

#[derive(Copy, Clone)]
pub struct ExtraForestSettings {
    pub n_estimators: usize,
    pub max_depth: MaxDepth,
    pub n_jobs: NJobs,
    pub bootstrap: bool,
}

impl Default for ExtraForestSettings {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            max_depth: MaxDepth::Infinite,
            n_jobs: NJobs::NoLimit,
            bootstrap: false,
        }
    }
}