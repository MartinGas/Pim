
use std::iter::IntoIterator;
use std::collections::HashMap;
use std::cmp::{Ord, Eq};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use std::cell::RefCell;
use std::rc::Rc;

use bit_set::BitSet;

mod linked_trie;

pub type Transaction = BitSet;
pub type Query = Vec<Item>;
pub type Count = u64;
pub type Item = usize;
pub type ItemVec = Vec<Item>;

/// Reads data in FIMI format into a data base
pub fn read_data( path: &str ) -> Result<LinkedTrieBackedDatabase, String> {
    let path = Path::new( path );
    let file = File::open( path ).map_err( |e| e.to_string() )?;
    let reader = BufReader::new( file );
    let data: Vec<ItemVec> = reader.lines()
        .filter_map( |l| l.ok() )
        .filter_map( |l| read_transaction( &l, " " ))
	.collect();
    let mut database = LinkedTrieBackedDatabase::new( &data );
    database.add( &data );
    Ok( database )
}

fn read_transaction( line: &str, splitter: &str ) -> Option<ItemVec> {
    let mut transaction = ItemVec::new();
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
	Con: IntoIterator<Item = &'a ItemVec>;

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


type ItemBuffer = Rc<RefCell<Vec<Item>>>;

pub struct LinkedTrieSequenceIterator<'a> {
    /// access to the buffered sequence
    shared_sequence: ItemBuffer,
    /// internal trie iterator
    iterator: linked_trie::TrieIterator<'a>,
    /// special greatest item that marks the end of a sequence
    stop_item: Item,
}

struct SharedSequenceBuffer {
    shared_sequence: ItemBuffer,
    pop_stack: Vec<usize>,
}

impl Database for LinkedTrieBackedDatabase {

    fn add <'a, Con> ( &mut self, transactions: Con ) where
	Con: IntoIterator<Item = &'a ItemVec>
    {
	for t in transactions.into_iter() {
	    let mut mapped_transaction: Vec<Item> = self.map_into( t );
	    // add marker so we know where a transaction stops in the trie
	    mapped_transaction.push( self.stop_item );
	    self.data.add( mapped_transaction );
	};
    }

    fn query_support( &self, query: Query ) -> Count {
	// let us use the mapped items
	// let mapped_query: Vec<Item> = self.map_into( &query );
	self.data.query_support( query )
    }
}

impl <'a> IntoIterator for &'a LinkedTrieBackedDatabase {
    type Item = (Transaction, Count);
    type IntoIter = LinkedTrieSequenceIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
	let buffer: ItemBuffer = Rc::new( RefCell::new( Vec::new() ));
	let consumer = SharedSequenceBuffer::new( buffer.clone() );
	LinkedTrieSequenceIterator::new( self.data.iterate_with_consumer( consumer ), buffer, self.stop_item )
    }
}

impl LinkedTrieBackedDatabase {

    // \todo rename into new_with_dynamic_order
    /// Initializes the data base with a sample that serves to determine the item order.
    pub fn new <'a, D, T> ( sample: D ) -> LinkedTrieBackedDatabase where
	D: IntoIterator<Item = T>,
	T: IntoIterator<Item = &'a Item>
    {
	let item_stream = sample.into_iter().flat_map( |t| t.into_iter() );
	let frequency_map: HashMap<Item, u64> = calc_item_frequencies( item_stream );

	println!( "item map {:?}", map_by_score( &frequency_map ) );
	
	LinkedTrieBackedDatabase {
	    data: linked_trie::Trie::new(),
	    // singleton_counts: HashMap::new(),
	    item_map: map_by_score( &frequency_map ),
	    stop_item: frequency_map.len(), // stop at max_item + 1
	}
    }

    /// Initialize the data base with a fixed item order
    pub fn new_with_static_order( universe: &[Item] ) -> LinkedTrieBackedDatabase {
	let stop_item = universe.iter().max().map( |item| *item ).unwrap_or( 0 ) + 1;
	LinkedTrieBackedDatabase {
	    data: linked_trie::Trie::new(),
	    item_map: universe.iter().map( |item| (*item, *item) ).collect(),
	    stop_item
	}
    }

    /// Maps items into internal representation and returns the result.
    fn map_into <'a, I> ( &self, items: I ) -> Vec<Item> where I: IntoIterator<Item = &'a Item> {
	items.into_iter()
	    .map( |i| *self.item_map.get( i ).expect( "Have mapping for item" ))
	    .collect()
    }
}

impl <'a> Iterator for LinkedTrieSequenceIterator<'a> {
    type Item = (Transaction, Count);

    fn next(&mut self) -> Option<Self::Item> {
	// keep visiting until arriving at a terminal node or running out of nodes
	let mut new_chunk = self.iterator.next();
	while new_chunk.is_some() {
	    if let Some( transaction ) = self.produce() {
		let (_, count) = new_chunk.expect( "checked is some" );
		return Some( (transaction, count) );
	    }
	    new_chunk = self.iterator.next();
	}
	None
    }
}

impl <'a> LinkedTrieSequenceIterator<'a> {

    fn new( trie_iterator: linked_trie::TrieIterator<'a>, buffer: ItemBuffer, stop_item: Item ) -> LinkedTrieSequenceIterator<'a> {
	LinkedTrieSequenceIterator{
	    shared_sequence: buffer,
	    iterator: trie_iterator,
	    stop_item,
	}
    }

    pub fn produce( &self ) -> Option<Transaction> {
	let sequence = self.shared_sequence.borrow();
	if let Some( (stop, content) ) = sequence.split_last() {
	    if *stop == self.stop_item {
		let greatest_item = content.last().map_or_else( || 0, |itm| *itm );
		let mut transaction = BitSet::with_capacity( greatest_item );
		for item in content {
		    transaction.insert( *item );
		}
		Some( transaction )
	    } else { None }
	} else { None }
    }
}

impl SharedSequenceBuffer {

    fn new( buffer: ItemBuffer ) -> SharedSequenceBuffer {
	SharedSequenceBuffer{
	    shared_sequence: buffer,
	    pop_stack: vec!( 0 ), // root added nothing
	}
    }
}

impl <'a, L> linked_trie::Consumer<L> for SharedSequenceBuffer where L: IntoIterator<Item = &'a Item> {

    fn enter( &mut self, label: L ) {
	// push all elements from label
	let mut sequence = self.shared_sequence.borrow_mut();
	let len_before = sequence.len();
	for item in label {
	    sequence.push( *item );
	}
	let delta = sequence.len() - len_before;
	self.pop_stack.push( delta );
    }

    fn leave( &mut self ) {
	// remove as many items as were added last
	let num_items_last_added = self.pop_stack.pop().expect( "Add whenever enter. Leave once for every add." );
	let mut sequence = self.shared_sequence.borrow_mut();
	for _ in 0 .. num_items_last_added {
	    sequence.pop();
	}
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

#[cfg(test)]
mod test {

    use super::*;
    
    /// Compresses some sequences into a trie. Visits all sequences afterwards.
    #[test]
    fn test_iteration() {

	// need data that has items in descending frequency order already
	let data = vec!(
	    vec!( 0, 1, 2, 3, 4 ),
	    vec!( 0, 1, 2, 3 ),
	    vec!( 0, 1, 2 ),
	    vec!( 0, 1 ),
	    vec!( 0, 1 ), // duplicate
	    vec!( 0 ),
	);

	type ItemVec = Vec<Item>;
	let mut expectations: HashMap<ItemVec, Count> = HashMap::new();
	expectations.insert( vec!( 0 ), 1 );
	expectations.insert( vec!( 0, 1 ), 2 );
	expectations.insert( vec!( 0, 1, 2 ), 1 );
	expectations.insert( vec!( 0, 1, 2, 3 ), 1 );
	expectations.insert( vec!( 0, 1, 2, 3, 4 ), 1 );

	let mut database = LinkedTrieBackedDatabase::new( &data );
	database.add( &data );

	for (real_chunk, real_count) in database.into_iter() {
	    let real_chunk_vector: ItemVec = real_chunk.iter().collect();
	    let count = expectations.remove( &real_chunk_vector );

	    assert!( count.is_some() );
	    assert_eq!( count.unwrap(), real_count );
	}
	assert!( expectations.is_empty() );
    }
}
