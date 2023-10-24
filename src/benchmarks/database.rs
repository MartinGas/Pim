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
    let mut database = LinkedTrieBackedDatabase::new( &data );
    database.add( &data );

    info!( "Start benchmark: uniform queries" );
    let n = 10000;
    let time = benchmark_uniform_queries( &database, n );
    info!( "Result: {n} uniform queries took {}ms", time.as_millis() );

    Result::Ok( () )
}




fn benchmark_uniform_queries( database: &LinkedTrieBackedDatabase, number_queries: u64 ) -> Duration {
    let mut universe = database.create_universe();
    let m = universe.len();
    // There are more distinct queries around m/2 length.
    // Use binomial, to obtain uniform distribution over queries.
    // let length_distribution = Binomial::new( 0.5, m as u64 ).unwrap();
    // Use uniform to give shorter sequences a shot too
    let length_distribution = DiscreteUniform::new( 1, m as i64 ).unwrap();

    let mut query_time = Duration::new( 0, 0 );
    let number_buckets = 10;
    let mut query_time_buckets = vec!( Duration::new( 0, 0 ); number_buckets );

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
    debug!( "time by length {length_query_times:?} [ms]" );
    query_time
}

fn generate_random_query( universe: &mut Itemvec, length: usize ) -> Itemvec {
    let mut dist = DiscreteUniform::new( 0, length as i64 - 1 ).unwrap();
    let mut gen = thread_rng();
    let sample_length = dist.sample( &mut gen ) as usize;

    let mut query = Itemvec::new();
    for sample_count in 0 .. sample_length {
	dist = DiscreteUniform::new( sample_count as i64, sample_length as i64 ).unwrap();
	let i = dist.sample( &mut gen ) as usize;
	query.push( universe[i] );
	// move i into sample count place to avoid drawing it again
	universe.swap( sample_count, i );
    }
    query
}

fn prepare_logging() {
    let tracer = tracing_subscriber::fmt::fmt()
        .with_max_level( tracing_subscriber::filter::LevelFilter::DEBUG )
        .finish();
    tracing::subscriber::set_global_default( tracer ).unwrap();
}
