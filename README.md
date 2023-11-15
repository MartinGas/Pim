# Pim
Probabilistic Itemset Mining

## What is it?
Pim is a tool for informative pattern mining. It summarizes a binary transactional dataset with a succinct set of conjunctive patterns. The miner reports the patterns that optimize its probabilistic, generative model. Although operational, the miner is still in the earliest stages of development.

## How to use?
You will need to compile the project from its source, which requires Rust installation with Cargo.
To build, enter the project's root directory and
```
cargo build --bin miner 
```
Pim mines datasets in FIMI format. A file in FIMI format contains one transaction per line and every transaction consists of space separated numbers, which represent its items.
Pim produces a JSON file that contains the optimized model, which includes the patterns.
Run Pim with Cargo to produce output.json on data.fimi:
```
cargo run --bin miner -- -o output.json data.fimi
```
