use std::marker::Copy;

#[derive(Copy, Clone, Debug)]
pub enum NJobs {
    NoLimit,
    Value(usize)
}

#[derive(Copy, Clone, Debug)]
pub struct ExtraForestSettings {
    pub n_estimators: usize,
    pub n_jobs: NJobs,
    pub bootstrap: bool,
}

impl Default for ExtraForestSettings {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            n_jobs: NJobs::NoLimit,
            bootstrap: false,
        }
    }
}