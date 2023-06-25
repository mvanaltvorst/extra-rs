use ndarray::{ArrayBase, Ix1, OwnedRepr};
use crate::tree::branch::Branch;
use crate::tree::leaf::Leaf;

pub trait Node<T> 
where T: Copy {
    fn predict(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T;
}

impl<T: Copy> Node<T> for Leaf<T> {
    fn predict(&self, _x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T {
        self.prediction
    }
}

impl<T: Copy> Node<T> for Branch<T> {
    fn predict(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T {
        if self.splitter.split(x) {
            self.right.predict(x)
        } else {
            self.left.predict(x)
        }
    }
}
