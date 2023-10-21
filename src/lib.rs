
pub mod data;
pub mod miner;
pub mod model;

pub use data::{Item, Transaction, Count, DataPair, Database};
pub use model::Model;
pub use miner::Miner;

fn library() {
    println!( "This is fromt he library" );
}
