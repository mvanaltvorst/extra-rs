pub trait Splitter {
    pub fn split(&self, x: TreeRow) -> bool;
}

pub struct NumericalSplitter {
    attribute_index: f32,
    pivot: f32,
}

pub struct CategoricalSplitter {
    attribute_index: f32,
    subset: HashSet<
}