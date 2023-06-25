use ndarray::{ArrayBase, Ix1, Ix2, Data};

// T is either f32 or bool.
pub struct TreeDataset<T>
where T: Data<Elem = f32> + Data<Elem = bool> { 
    pub X: ArrayBase<T, Ix2>,
    pub y: ArrayBase<T, Ix1>,
}
