use crate::data::tree_row::TreeRow;

// T is either f32 or bool.
pub struct TreeDataset<T>
where T: Copy { 
    pub X: Vec<TreeRow>,
    pub y: Vec<T>,
}
