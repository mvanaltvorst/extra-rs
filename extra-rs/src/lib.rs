#![feature(test)]

extern crate test;

pub mod data;
pub mod extra_forest;
pub mod extra_tree;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use polars::prelude::*;
    use test::Bencher;

    // #[test]
    // fn it_works() {
    //     assert_eq!(4, add_two(2));
    // }

    // #[bench]
    // fn bench_train_tree(b: &mut Bencher) {
    //     // First we read the data
    //     // Python code:
    //     // N = 100000
    //     // df = pd.read_parquet('data.parquet').head(N)
    //     // df_train, df_test = df.iloc[:int(0.8 * N)], df.iloc[int(0.8 * N):]
    //     // X_train = df_train.drop(columns=['y'])
    //     // y_train = df_train['y']
    //     // X_test = df_test.drop(columns=['y'])
    //     // y_test = df_test['y']
    //     // the package we use to read .parquet
    //     // is "parquet-rs"
    //     const N: usize = 100_000;
    //     let mut file = std::fs::File::open(
    //         "/Users/maurits/Downloads/code_to_read/ert/extra-rs/benchmarks/data.parquet",
    //     )
    //     .unwrap();

    //     let df = ParquetReader::new(&mut file).finish().unwrap();

    //     let df_train = df.head(Some((0.8 * (N as f32)) as usize));
    //     let df_test = df.tail(Some((0.2 * (N as f32)) as usize));

    //     let y_train = df_train.select(&["y"]).unwrap();
    //     let y_test = df_test.select(&["y"]).unwrap();
    //     let X_train = df_train.drop("y").unwrap();
    //     let X_test = df_test.drop("y").unwrap();

    //     let train_dataset = data::tree_dataset::TreeDataset {
    //         X: X_train.to_ndarray::<Float32Type>(IndexOrder::C).unwrap(),
    //         y: y_train
    //             .to_ndarray::<Float32Type>(IndexOrder::C)
    //             .unwrap()
    //             .slice(s![.., 0])
    //             .to_owned(),
    //     };

    //     b.iter(|| {
    //         let mut model = extra_tree::extra_tree_regressor::ExtraTreeRegressor::new(
    //             super::extra_tree::extra_tree_settings::ExtraTreeSettings::new(
    //                 super::extra_tree::extra_tree_settings::MaxFeatures::Sqrt,
    //                 2,
    //                 false,
    //                 super::extra_tree::extra_tree_settings::MaxDepth::Infinite,
    //             ),
    //         );
    //         model.build(&train_dataset);
    //     });
    // }
}
