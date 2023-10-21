
pub mod data;
pub mod miner;
pub mod model;

use tracing::*;

pub use data::{Item, Transaction, Count, DataPair, Database};
pub use model::Model;
pub use miner::Miner;

/// Objects that can be recorded in the log
pub trait Loggable {
    fn log(&self, message: &str, level: tracing::Level );
}
