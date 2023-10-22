
pub mod data;
pub mod miner;
pub mod model;
pub mod io;

use tracing::*;

pub use data::{Item, Transaction, Count, DataPair, Database};
pub use model::Model;
pub use miner::Miner;

/// Used as intermediate representation for itemset patterns
pub type Itemvec = Vec<Item>;

/// Objects that can be recorded in the log
pub trait Loggable {
    fn log(&self, message: &str, level: tracing::Level );
}
