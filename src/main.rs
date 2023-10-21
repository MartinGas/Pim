
// mod data;
// mod model;
// mod miner;

// use transmine::data::{self, DataPair, Item, Transaction, Count};
// use transmine::model::Model;
// use transmine::miner::Miner;
use transmine::*;
use data::read_data;

fn main() -> Result<(), String>{

    let database = read_data( "./data/penguins/penguins.fimi" )?;
    // let database = data::read_data( "./data/census/census.fimi" )?;
    let universe: Vec<Item> = database.create_universe();
    let mut model = model::BernoulliAssignment::new( universe.iter() );
    let mut miner = miner::EmMiner::new( 10 );
    miner.mine( &database, &mut model );

    Ok( () )
}
