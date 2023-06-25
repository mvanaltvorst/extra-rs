use ndarray::{ArrayBase, Ix1, OwnedRepr};


pub enum Splitter {
    NumericalSplitter(usize, f32), // attribute_index, pivot
}

impl Splitter {
    pub fn split(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> bool {
        // False is left, true is right.
        match self {
            Splitter::NumericalSplitter(attribute_index, pivot) => x[*attribute_index] <= *pivot,
        }
    }
}
