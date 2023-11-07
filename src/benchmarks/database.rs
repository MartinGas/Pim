use tracing::{info,debug};
use tracing_subscriber;

use rand::prelude::*;
use statrs::distribution::{DiscreteUniform, Binomial};

use std::time::*;

use transmine::*;
use transmine::data::{self, LinkedTrieBackedDatabase};

fn main() -> Result<(), String> {
    prepare_logging();

    let data = io::read_data(  "./data/census/census.fimi", |line| io::parse_fimi_to_vec( line, " " ))?;
    let data: Vec<Itemvec> = data.collect();
    // let frequencies = pre::calculate_item_frequency( &data );
    let mut database = LinkedTrieBackedDatabase::new_with_frequency_order( &data );
    database.add( &data );

    let n = 5000;

    info!( "Start benchmark: queries without cache" );
    database.set_max_cache_length( 0 );
    let time = benchmark_uniform_queries( &database, n );
    info!( "Result: {n} uniform queries took {}ms", time.as_millis() );

    info!( "Start benchmark: queries with cache" );
    database.set_max_cache_length( 10 ); // cache all queries
    let time = benchmark_uniform_queries( &database, n );
    info!( "Result: {n} uniform queries took {}ms", time.as_millis() );

    Result::Ok( () )
}




fn benchmark_uniform_queries( database: &LinkedTrieBackedDatabase, number_queries: u64 ) -> Duration {
    let mut universe = database.create_universe();
    println!( "universe {universe:?}" );
    
    let m = universe.len();
    // There are more distinct queries around m/2 length.
    // Use binomial, to obtain uniform distribution over queries.
    // let length_distribution = Binomial::new( 0.5, m as u64 ).unwrap();
    // Use uniform to give shorter sequences a shot too
    let length_distribution = DiscreteUniform::new( 1, m as i64 ).unwrap();

    let mut query_time = Duration::new( 0, 0 );
    let number_buckets = 10;
    let mut query_time_buckets = vec!( Duration::new( 0, 0 ); number_buckets );
    let mut query_count_buckets = vec!( 0; number_buckets );

    for _ in 0 .. number_queries {
	let mut gen = thread_rng();
	let query_length = length_distribution.sample( &mut gen ) as usize;
	let query = generate_random_query( &mut universe, query_length );

	let start = Instant::now();
	database.query_support( query );
	let time_spent = Instant::now().duration_since( start );
	query_time += time_spent;
	debug!( "bucket index {number_buckets} * {query_length} / {m} ");
	let bucket_index = number_buckets * (query_length - 1) / m;

	query_time_buckets[ bucket_index ] += time_spent;
    }

    let length_query_times: Vec<u64> = query_time_buckets.iter().map( |d| d.as_millis() as u64 ).collect();
    info!( "time by length {length_query_times:?} [ms]" );
    query_time
}

fn generate_random_query( universe: &mut Itemvec, length: usize ) -> Itemvec {
    let m = universe.len() as i64;
    let mut gen = thread_rng();

    let mut query = Itemvec::new();
    for sample_count in 0 .. length {
	let item_dist = DiscreteUniform::new( sample_count as i64, m - 1 as i64 ).unwrap();
	let i = item_dist.sample( &mut gen ) as usize;
	query.push( universe[i] );
	// move i into sample count place to avoid drawing it again
	universe.swap( sample_count, i );
    }
    query.sort(); // canonical representation
    query
}

fn prepare_logging() {
    let tracer = tracing_subscriber::fmt::fmt()
        .with_max_level( tracing_subscriber::filter::LevelFilter::INFO )
        .finish();
    tracing::subscriber::set_global_default( tracer ).unwrap();
}
