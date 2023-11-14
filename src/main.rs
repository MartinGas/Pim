use tracing;
use tracing_subscriber;
use clap::{self, Parser};

use transmine::*;
// use data::*;
use io::{read_data, parse_fimi_to_vec, write_model};

#[derive(Parser)]
struct Arguments {
    /// data set to mine
    #[arg()]
    data_path: String,
    /// Run with debug logging
    #[arg( long, default_value_t = false )]
    debug: bool,
    /// optional path to write the model to
    #[arg()]
    out_path: Option<String>,
}

type UseModel = model::BernoulliAssignment;
type UseDb = data::DefaultDb;

fn setup_logging( args: &Arguments ) -> Result<(), String> {
    let tracer = tracing_subscriber::fmt::fmt();
    let tracer = if args.debug {
	tracer.with_max_level( tracing_subscriber::filter::LevelFilter::TRACE )
    } else {
	tracer.with_max_level( tracing_subscriber::filter::LevelFilter::INFO )
    };
    let tracer = tracer.finish();
    tracing::subscriber::set_global_default( tracer ).map_err( |err| err.to_string() )
}

fn populate_database( _args: &Arguments, data: Vec<Itemvec> ) -> UseDb {
    let mut database = data::LinkedTrieDatabaseBuilder::new( 0 );
    database.remap_by_frequency( data.iter() );
    // let mut database = database.build_with_edgelist();
    let mut database = database.build_with_edgelist_better();
    database.set_max_cache_length( 0 ); // no cache
    database.add( &data );
    database
}

fn initialize_model( _args: &Arguments, database: &data::DefaultDb  ) -> UseModel {
    let universe: Vec<Item> = database.create_universe();
    model::BernoulliAssignment::new( universe.iter() )
}

fn initialize_miner( _args: &Arguments ) -> miner::EmMiner<UseDb, UseModel> {
    let mut miner = miner::EmMiner::new( 1000 );
    let mut formatter = model::BernoulliFormatter::new();
    formatter.show_patterns();
    formatter.show_items();
    miner.provide_model_formatter( formatter );
    miner
}

fn finalize_results( args: &Arguments, model: &UseModel ) -> Result<(), String> {
    if let Some( path ) = &args.out_path {
	write_model( model, path.as_str() )
    } else {
	Result::Ok( () )
    }
}


fn main() -> Result<(), String>{
    let args = Arguments::parse();
    setup_logging( &args )?;
    let data: Vec<Itemvec> = read_data( &args.data_path, |line| parse_fimi_to_vec( line, " " ))?.collect();
    let database = populate_database( &args, data );
    let mut model = initialize_model( &args, &database );
    let mut miner = initialize_miner( &args );
    miner.mine( &database, &mut model );
    finalize_results( &args, &model )
}
