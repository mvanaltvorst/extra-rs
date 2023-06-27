use ndarray::{ArrayBase, Ix1, Ix2, Data, OwnedRepr};

// X is f32.
// y is f32 or bool.
pub struct TreeDataset<T> where T: Copy { 
    pub X: ArrayBase<OwnedRepr<f32>, Ix2>,
    pub y: ArrayBase<OwnedRepr<T>, Ix1>,
}
