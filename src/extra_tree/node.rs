use ndarray::{ArrayBase, Ix1, OwnedRepr};
use crate::extra_tree::splitter::Splitter;

// pub trait Node<T> 
// where T: Copy {
//     fn predict(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T;
// }

// impl<T: Copy> Node<T> for Leaf<T> {
//     fn predict(&self, _x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T {
//         self.prediction
//     }
// }

// impl<T: Copy> Node<T> for Branch<T> {
//     fn predict(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> T {
//         if self.splitter.split(x) {
//             self.right.predict(x)
//         } else {
//             self.left.predict(x)
//         }
//     }
// }

pub enum Node {
    Leaf(f32),
    Branch(Splitter, Box<Node>, Box<Node>),
}

impl Node {
    pub fn predict(&self, x: &ArrayBase<OwnedRepr<f32>, Ix1>) -> f32 {
        match self {
            Node::Leaf(prediction) => *prediction,
            Node::Branch(splitter, left, right) => {
                if splitter.split(x) {
                    right.predict(x)
                } else {
                    left.predict(x)
                }
            }
        }
    }
}