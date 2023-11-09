use tracing;
use tracing_subscriber;

use transmine::*;
use data::*;
use io::{read_data, parse_fimi_to_vec};

fn main() -> Result<(), String>{
    let tracer = tracing_subscriber::fmt::fmt()
        .with_max_level( tracing_subscriber::filter::LevelFilter::INFO )
	.finish();
    tracing::subscriber::set_global_default( tracer ).map_err( |err| err.to_string() )?;

    // let data = read_data( "./data/penguins/penguins.fimi", |line| parse_fimi_to_vec( line, " " ) )?;
    let data = read_data( "./data/census/census.fimi", |line| parse_fimi_to_vec( line, " " ) )?;
    let data: Vec<Itemvec> = data.collect();

    let mut database = LinkedTrieDatabaseBuilder::new( 0 );
    database.remap_by_frequency( data.iter() );
    let mut database = database.build_with_edgelist();
    database.add( &data );
    
    let universe: Vec<Item> = database.create_universe();
    let mut model = model::BernoulliAssignment::new( universe.iter() );
    let mut miner = miner::EmMiner::new( 1000 );

    let mut formatter = model::BernoulliFormatter::new();
    formatter.show_patterns();
    miner.provide_model_formatter( formatter );
    miner.mine( &database, &mut model );

    Ok( () )
}
