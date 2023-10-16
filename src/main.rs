
mod data;
mod model;
mod miner;

pub use data::{Database, DataPair, Item, Transaction, Count};
pub use model::Model;
pub use miner::Miner;

fn main() -> Result<(), String>{

    let database = data::read_data( "./data/census/census.fimi" )?;
    let universe: Vec<Item> = database.create_universe();
    let mut model = model::BernoulliAssignment::new( universe.iter() );
    let mut miner = miner::EmMiner::new( 10 );
    miner.mine( &database, &mut model );

    Ok( () )
}
