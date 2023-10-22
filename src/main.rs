
use tracing;
use tracing_subscriber;

use transmine::*;
use data::populate_trie_database;
use io::{read_data, parse_fimi_to_vec};

fn main() -> Result<(), String>{
    let tracer = tracing_subscriber::fmt::fmt()
        .with_max_level( tracing_subscriber::filter::LevelFilter::DEBUG )
	.finish();
    tracing::subscriber::set_global_default( tracer ).map_err( |err| err.to_string() )?;

    let data = read_data( "./data/penguins/penguins.fimi", |line| parse_fimi_to_vec( line, " " ) )?;
    // let database = data::read_data( "./data/census/census.fimi" )?;

    let database = populate_trie_database( data );
    let universe: Vec<Item> = database.create_universe();
    let mut model = model::BernoulliAssignment::new( universe.iter() );
    let mut miner = miner::EmMiner::new( 10 );

    let mut formatter = model::BernoulliFormatter::new();
    formatter.show_patterns();
    miner.provide_model_formatter( formatter );
    miner.mine( &database, &mut model );

    Ok( () )
}
