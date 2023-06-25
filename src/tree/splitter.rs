use ndarray::{ArrayBase, Ix1, OwnedRepr};

pub trait Splitter {
    fn split(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> bool;
}

// We only support numerical splits (numerical data) for now
pub struct NumericalSplitter {
    attribute_index: usize,
    pivot: f32,
}

impl Splitter for NumericalSplitter {
    fn split(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> bool {
        x[self.attribute_index] < self.pivot
    }
}

impl NumericalSplitter {
    pub fn new(attribute_index: usize, pivot: f32) -> Self {
        Self {
            attribute_index,
            pivot,
        }
    }
}