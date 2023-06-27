use ndarray::{ArrayBase, Ix1, Data};


pub enum Splitter {
    NumericalSplitter(usize, f32), // attribute_index, pivot
}

impl Splitter {
    pub fn split<T>(&self, x: &ArrayBase<T, Ix1>) -> bool
    where T: Data<Elem = f32> {
        // False is left, true is right.
        match self {
            Splitter::NumericalSplitter(attribute_index, pivot) => x[*attribute_index] <= *pivot,
        }
    }
}
