use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr, Array1, Axis, Data};
use rand::Rng;
use rand::seq::SliceRandom;
use crate::extra_tree::splitter::Splitter;
use crate::extra_tree::extra_tree_settings::{ExtraTreeSettings, MaxDepth};
use crate::data::tree_dataset::TreeDataset;

use super::extra_tree_settings::MaxFeatures;
use super::node::Node;


pub fn pick_random_split(samples: ArrayBase<OwnedRepr<f32>, Ix1>, attribute_index: usize) -> Splitter {
    let (min, max) = samples.iter().fold((f32::MAX, f32::MIN), |(min, max), &x| {
        (min.min(x), max.max(x))
    });
    let pivot = rand::thread_rng().gen::<f32>() * (max - min) + min;
    Splitter::NumericalSplitter(attribute_index, pivot)
}

pub fn is_constant<T>(X: &ArrayBase<T, Ix2>) -> Array1<bool> 
where T: Data<Elem = f32> {
    // Returns a binary array for each column of X
    // indicating whether the column is a constant.
    X
    .axis_iter(Axis(1))
    .map(|vs| vs.iter().all(|&v| v == vs[0]))
    .collect()
}

pub fn stop_expansion<T: Copy + PartialEq>(
    dataset: &TreeDataset<T>,
    is_constant: &ArrayBase<OwnedRepr<bool>, Ix1>,
    settings: &ExtraTreeSettings,
    current_depth: usize,
) -> bool {
    // cannot exceed `max_depth`
    if let MaxDepth::Value(k) = settings.max_depth {
        if current_depth >= k {
            return true;
        }
    }

    // should be at least `min_samples_split` in the dataset.
    if dataset.y.len() < settings.min_samples_split {
        return true;
    }

    // if no variables anymore
    if is_constant.iter().all(|&x| x) {
        return true;
    }

    // if y is constant
    if dataset.y.iter().all(|&v| v == dataset.y[0]) {
        return true;
    }

    false
}

pub fn split_sample<T: Copy>(
    splitter: &Splitter,
    dataset: &TreeDataset<T>,
) -> (
    TreeDataset<T>,
    TreeDataset<T>,
) {
    match *splitter {
        Splitter::NumericalSplitter(attribute_index, pivot) => {
            let samples = dataset.X.column(attribute_index);

            let (left_indices, right_indices) = samples.iter().enumerate().fold(
                (Vec::new(), Vec::new()),
                |(mut left_indices, mut right_indices), (i, &x)| {
                    if x > pivot {
                        right_indices.push(i);
                    } else {
                        left_indices.push(i);
                    }
                    (left_indices, right_indices)
                },
            );

            let left = TreeDataset { 
                X: dataset.X.select(Axis(0), &left_indices),
                y: dataset.y.select(Axis(0), &left_indices),
            };
            let right = TreeDataset { 
                X: dataset.X.select(Axis(0), &right_indices),
                y: dataset.y.select(Axis(0), &right_indices),
            };

            (left, right)
        }
    }
}

pub fn create_subtree<T: Copy + PartialEq + Into<f32>>(score: fn(&Splitter, &TreeDataset<T>) -> f32, dataset: &TreeDataset<T>, settings: &ExtraTreeSettings, current_depth: usize) -> Node {
    let is_constant = is_constant(&dataset.X);
    if stop_expansion(dataset, &is_constant, settings, current_depth) {
        let prediction: f32 = dataset
            .y
            .iter()
            .map(|&v| v.into())
            .sum::<f32>()
            / (dataset.y.len() as f32);
        Node::Leaf(prediction)
    } else {
        // In case of automatic max features in a classification problem,
        // we take the length of the dataset as number of splits to consider.
        // TODO: reference
        let k = match settings.max_features {
            MaxFeatures::Sqrt => (dataset.y.len() as f32).sqrt().ceil() as usize,
            MaxFeatures::Value(k) => k,
        };

        // take k random features without replacement.
        let rand_indices = {
            let mut indices: Vec<usize> = (0..is_constant.len())
                .filter(|&i| !is_constant[i])
                .collect();

            let mut rng = rand::thread_rng();
            // TODO: figure out how to do this without
            // shuffling the entire list.
            indices.shuffle(&mut rng);
            indices.iter().take(k.min(indices.len())).cloned().collect::<Vec<usize>>()
        };

        let X_feature_subset = dataset.X.select(Axis(1), &rand_indices);

        let best_split = (0..rand_indices.len())
            .map(|i| pick_random_split(X_feature_subset.index_axis(Axis(1), i).to_owned(), rand_indices[i]))
            .map(|splitter| (score(&splitter, dataset), splitter))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1;

        let (left, right) = split_sample(&best_split, dataset);

        Node::Branch(
            best_split,
            Box::new(create_subtree(score, &left, settings, current_depth + 1)),
            Box::new(create_subtree(score, &right, settings, current_depth + 1)),
        )
    }
}