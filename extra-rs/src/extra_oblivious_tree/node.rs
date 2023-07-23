use ndarray::{ArrayBase, Ix1, Data};
use crate::extra_tree::splitter::Splitter;


#[derive(Debug)]
pub enum Node {
    Leaf(f32),
    Branch(Splitter, Box<Node>, Box<Node>),
    Unexplored
}

impl Node {
    pub fn predict<T>(&self, x: &ArrayBase<T, Ix1>) -> f32
    where T: Data<Elem = f32> {
        match self {
            Node::Leaf(prediction) => *prediction,
            Node::Branch(splitter, left, right) => {
                if splitter.split(x) {
                    right.predict(x)
                } else {
                    left.predict(x)
                }
            },
            Node::Unexplored => unreachable!("Node should be explored before predicting.")
        }
    }
}