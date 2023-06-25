use crate::tree::node::Node;
use crate::tree::splitter::Splitter;

pub struct Branch<T>
where T: Copy {
    pub left: Box<dyn Node<T>>,
    pub right: Box<dyn Node<T>>,
    pub splitter: dyn Splitter
}