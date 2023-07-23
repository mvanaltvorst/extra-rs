use super::extra_tree_settings::MaxDepth;
use super::extra_tree_regressor::ExtraTreeRegressor;
use super::splitter::Splitter;
use super::node::Node;

pub struct ExtraTreeRegressorInferencer {
    // We avoid jumping to random memory locations for the extra tree inferencer
    // by firstly calculating each of the jumps and later jumping to the correct
    // leaf node that is stored in a contiguous array.

    // We assume we have a max_depth of 8, so we have 2^8 = 256 leaf nodes.
    // Similarly, we assume a maximum number of 255 features to ensure our feature_index
    // fits into a u8.
    attribute_indices: [u8; 1 << 8],
    pivots: [u8; 1 << 8],
    leaf_values: [u8; 1 << 8],
}

fn get_masked_leaf_index(leaf_index: usize, depth: usize) -> usize {
    // mask at depth = 0 is 0b10000000
    // mask at depth = 1 is 0b11000000
    // mask at depth = 2 is 0b11100000
    let mask = (0xff << (8 - depth)) & 0xff; 

    // we add a trailing 1 bit to encode the current depth into the state
    (leaf_index & mask) + (1 << (7 - depth))
}

pub fn fast_sigmoid(x: f32) -> u8 {
    // must be monotonic
    // -inf -> 0
    // 0 -> 128
    // inf -> 255

    // We use the fast sigmoid 256 * x / (1 + abs(x))
    ((x / (1.0 + x.abs())) * 255.0) as u8
}

pub fn inverse_fast_sigmoid(x: u8) -> f32 {
    // must be monotonic
    // 0 -> -inf
    // 128 -> 0
    // 255 -> inf

    // We use the fast sigmoid x / (1 - x)
    x as f32 / (1.0 - x as f32 / 255.0)
}


impl ExtraTreeRegressorInferencer {
    pub fn new(extra_tree: &ExtraTreeRegressor) -> Result<ExtraTreeRegressorInferencer, String> {
        match extra_tree.settings.max_depth {
            MaxDepth::Infinite => return Err("ExtraTreeInferencer only supports max_depth <= 8.".to_string()),
            MaxDepth::Value(max_depth) => if max_depth > 8 {
                return Err("ExtraTreeInferencer only supports max_depth <= 8.".to_string());
            }
        }

        let mut attribute_indices = [0u8; 1 << 8];
        let mut pivots = [0u8; 1 << 8];
        let mut leaf_values = [0u8; 1 << 8];

        // we iterate over all leaf indices and calculate the feature index, threshold and leaf value
        for leaf_index in 0..256 {
            let mut current_node = &extra_tree.root;
            let mut depth = 0;

            while depth < 8 {
                let acc_leaf_index = get_masked_leaf_index(leaf_index, depth);

                // println!("Leaf index: {:08b}, acc_leaf_index at depth={}: {:08b}", leaf_index, depth, acc_leaf_index);

                match current_node {
                    Node::Leaf(leaf_value) => {
                        // We found the correct leaf node.
                        // We store the leaf value and calculate feature indices and thresholds
                        // in the next loop.
                        leaf_values[leaf_index] = fast_sigmoid(*leaf_value);
                        // println!("Leaf value: {}", leaf_value);
                        break
                    },
                    Node::Unexplored => return Err("Encountered unexplored node.".to_string()),
                    Node::Branch(Splitter::NumericalSplitter(attribute_index, pivot), left, right) => {
                        if *attribute_index > 255 {
                            return Err("Attribute index is too large. Make sure you have <=255 features.".to_string());
                        }
                        let attribute_index = *attribute_index as u8;
                        attribute_indices[acc_leaf_index] = attribute_index;
                        pivots[acc_leaf_index] = fast_sigmoid(*pivot);
                        // println!("Attribute index: {}, pivot: {}", attribute_index, pivot);
                        if leaf_index & (1 << (7 - depth)) > 0 {
                            // right child
                            current_node = right;
                        } else {
                            // left child
                            current_node = left;
                        }
                    }
                }
                depth += 1;
            }

            while depth < 8 {
                // We have to fill the remaining depth with the last attribute index and pivot
                let acc_leaf_index = get_masked_leaf_index(leaf_index, depth);

                // println!("Leaf index: {:08b}, acc_leaf_index at depth={}: {:08b} (OOB)", leaf_index, depth, acc_leaf_index);

                attribute_indices[acc_leaf_index] = 0;
                if leaf_index & (1 << depth) > 0 {
                    // Right child
                    // We set the pivot to -inf to ensure we always go to the right child   
                    // println!("Setting pivot to -inf");
                    pivots[acc_leaf_index] = 0;
                } else {
                    // Left child
                    // println!("Setting pivot to +inf");
                    pivots[acc_leaf_index] = 0xff;
                }
                depth += 1;
            }
        }
        Ok(ExtraTreeRegressorInferencer {
            attribute_indices,
            pivots,
            leaf_values,
        })
    }

    pub fn predict_quantized(&self, x: &[u8]) -> u8 {
        let mut acc_leaf_index: usize = 0;
        for depth in 0..8 {
            let idx = acc_leaf_index + (1 << (7 - depth));
            let attribute_index = self.attribute_indices[idx];
            // println!("Acc leaf index: {:08b}, depth: {}, attribute index: {}", idx, depth, attribute_index);
            let pivot = self.pivots[idx];
            if x[attribute_index as usize] > pivot {
                // Right child
                acc_leaf_index += 1 << (7 - depth);
            }
        }
        self.leaf_values[acc_leaf_index]
    }

    pub fn get_debug_tree_description(&self) -> String {
        format!("{:?}, {:?}", self.attribute_indices, self.pivots)
    }
}
