use crate::tree::node::Node;

pub struct Branch<T> {
    pub left: Box<dyn Node<T>>,
    pub right: Box<dyn Node<T>>,
    split: dyn Splitter
}