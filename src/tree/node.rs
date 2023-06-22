use crate::data::tree_row::TreeRow;
use crate::tree::branch::{RegressionBranch, ClassificationBranch};
use crate::tree::leaf::Leaf;

pub trait Node<T> 
where T: Copy {
    fn predict(&self, x: &TreeRow) -> T;
}

impl<T: Copy> Node<T> for Leaf<T> {
    fn predict(&self, _x: &TreeRow) -> T {
        self.prediction
    }
}

impl Node<f32> for RegressionBranch {
    fn predict(&self, x: &TreeRow) -> f32 {
        unimplemented!()
    }
}

impl Node<bool> for ClassificationBranch { 
    fn predict(&self, x: &TreeRow) -> bool {
        unimplemented!()
    }
}