# Benchmarks

This directory contains benchmarks for the `extra-rs` crate. We track training time, inference time, and MSE for each benchmark. We are primarily interested in the inference time.

## Methodology
Our dataset consists of 1 million samples with 50 features. Every element is normally distributed with mean 0 and standard deviation 1. The target is a linear function of the features with normally distributed noise added. The noise has mean 0 and standard deviation 0.1. The linear function can be found in `gen_data.ipynb`.

We use 80% of the sample for training and 20% for testing. Measurements are made on a 2021 MacBook Pro M1 with 16 GB of RAM using all 8 cores. The library is compiled in release mode using `maturin develop --release`.

## Results

| Name | Description | N samples | Training Time | Inference Time | MSE |
--- | --- | --- | --- | --- | ---
| extra-rs 0.1.1 | n_estimators=80, min_samples_split=4, max_depth=10 | 100,000 | 42.3 s | 13.4 ms | 22.51 |
| LGBM 3.3.2 | n_estimators=80, min_samples_split=4, max_depth=10 | 100,000 | 1.2 s | 18.3 ms | 32.26 |

## Extra forest inferencer
The inferencer is built for trees with a max depth of 8. It is built upon bitmasks and lookup tables. At the moment of writing, it is ~30% slower than the default object-based inferencer. 

I suspect the slowdown is caused due to cache misses. Oblivious trees are trees where each split only depends on the depth and not on the splits made so far. The lookup table such a tree requires is many times smaller. This is yet to be implemented.

Feature quantization using the fast sigmoid function ($\frac {x} {|x| + 1}$) seems to have a negative impact on performance.