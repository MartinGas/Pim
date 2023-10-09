
use std::iter::IntoIterator;
use std::collections::HashMap;
use std::cmp::{Ord, Eq};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};

mod linked_trie;

pub type Transaction = Vec<Item>;
pub type Query = Vec<Item>;
pub type Count = u64;
pub type Item = usize;

/// Reads data in FIMI format into a data base
pub fn read_data( path: &str ) -> Result<LinkedTrieBackedDatabase, String> {
    let path = Path::new( path );
    let file = File::open( path ).map_err( |e| e.to_string() )?;
    let reader = BufReader::new( file );
    let data: Vec<Transaction> = reader.lines()
        .filter_map( |l| l.ok() )
        .filter_map( |l| read_transaction( &l, " " ))
	.collect();
    let mut database = LinkedTrieBackedDatabase::new( &data );
    database.add( &data );
    Ok( database )
}

fn read_transaction( line: &str, splitter: &str ) -> Option<Transaction> {
    let mut transaction = Transaction::new();
    for chunk in line.split( splitter ) {
	match Item::from_str_radix( chunk, 10 ) {
	    Ok( item ) => transaction.push( item ),
	    Err( _ ) => return None,
	}
    }
    Some( transaction )
}

pub trait Database {
    /// Adds every transaction produced by the iterator to the database
    fn add <'a, Con> ( &mut self, transactions: Con ) where
	Con: IntoIterator<Item = &'a Transaction>;

    /// Returns the support of all transactions containing all items in the query
    fn query_support( &self, query: Query ) -> Count;
}

/// Stores data as a linked trie with elements ordered by decreasing support.
pub struct LinkedTrieBackedDatabase {
    data: linked_trie::Trie,
    // singleton_counts: HashMap<Item, u64>,
    item_map: HashMap<Item, Item>,
    /// special bogus item that is appended to every transaction to mark its end
    stop_item: Item,
}

impl Database for LinkedTrieBackedDatabase {

    fn add <'a, Con> ( &mut self, transactions: Con ) where
	Con: IntoIterator<Item = &'a Transaction>
    {
	for t in transactions.into_iter() {
	    let mut mapped_transaction: Vec<Item> = self.map_into( t );
	    // add marker so we know where a transaction stops in the trie
	    mapped_transaction.push( self.stop_item );
	    self.data.add( mapped_transaction );
	};
    }

    fn query_support( &self, query: Query ) -> Count {
	let mapped_query: Vec<Item> = self.map_into( &query );
	self.data.query_support( mapped_query )
    }
}

impl LinkedTrieBackedDatabase {

    /// Initializes the data base with a sample that serves to determine the item order.
    pub fn new <'a, D, T> ( sample: D ) -> LinkedTrieBackedDatabase where
	D: IntoIterator<Item = T>,
	T: IntoIterator<Item = &'a Item>
    {
	let item_stream = sample.into_iter().flat_map( |t| t.into_iter() );
	let frequency_map: HashMap<Item, u64> = calc_item_frequencies( item_stream );
	
	LinkedTrieBackedDatabase {
	    data: linked_trie::Trie::new(),
	    // singleton_counts: HashMap::new(),
	    item_map: map_by_score( &frequency_map ),
	    stop_item: frequency_map.len(), // stop at max_item + 1
	}
    }
    
    /// Maps items into internal representation and returns the result.
    fn map_into <'a, I> ( &self, items: I ) -> Vec<Item> where I: IntoIterator<Item = &'a Item> {
	items.into_iter()
	    .map( |i| *self.item_map.get( i ).expect( "Have mapping for item" ))
	    .collect()
    }
}

fn calc_item_frequencies <'a, It> ( items: It ) -> HashMap<Item, u64> where
    It: Iterator<Item = &'a Item>,
{
    let mut counts: HashMap<Item, u64> = HashMap::new();
    for item in items {
	match counts.get_mut( item ) {
	    Some( count ) => *count += 1,
	    None => {
		counts.insert( *item, 1 );
	    }
	}
    }
    counts
}

fn map_by_score <S: Ord + Eq> ( item_to_score: &HashMap<Item, S> ) -> HashMap<Item, Item> {
    let mut items: Vec<Item> = item_to_score.keys().map( |i| *i ).collect();
    let compare = |left: &Item, right: &Item| {
	let left_score = item_to_score.get( &left ).expect( "every item has a score" );
	let right_score = item_to_score.get( &right ).expect( "every item has a score" );
	left_score.cmp( right_score ).reverse()
    };
    items.sort_unstable_by( compare );

    // the index in the sorted sequence becomes the new representation of the item
    let mut mapping: HashMap<Item, Item> = HashMap::with_capacity( item_to_score.len() );
    for (index, item) in items.iter().enumerate() {
	mapping.insert( *item, index );
    }
    mapping
}
