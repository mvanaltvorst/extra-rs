# Extremely randomized forests

This package written in Rust implements extremely randomized forests by Geurts [1]. There are Python bindings available as well. How to use this library can be found in [the examples](https://github.com/mvanaltvorst/extra-rs/tree/main/examples).

The main goal of this project is to learn more about random forests and provide a fast multi-threaded implementation of extremely randomized forests. I hope to make use of SIMD instructions in the future to further improve throughput and decrease latency.

## WIP
- [ ] Implement extremely randomized tree and forest regressors in Rust
- [ ] Python bindings using `maturin` and `pyo3`
- [ ] Benchmarks
- [ ] Documentation
- [ ] SIMD
    - [ ] x86_64: AVX512
    - [ ] ARM: NEON
- [ ] Feature importance

## Installation
```bash
git clone https://github.com/mvanaltvorst/extra-rs.git
cd extra-rs
pip install maturin
maturin develop --release
```

## Architecture
`extra-rs` contains the Rust library which implements extremely randomized forests from scratch, whereas `extra-py` is the Maturin project that exposes Python bindings.

[1] Geurts, P., Ernst, D. & Wehenkel, L. Extremely randomized trees. Mach Learn 63, 3–42 (2006). https://doi.org/10.1007/s10994-006-6226-1