# Zk-Torch

## To run a provided simple example
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup override set nightly

cargo run --release --bin zk_torch --features fold -- config.yaml
