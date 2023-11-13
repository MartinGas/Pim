
use std::cmp::Ordering;

use bit_set::BitSet;
use bit_vec::BitVec;

use super::*;

// Improves upon the original edgelist by
// (a) avoiding hash maps
// (b) O(1) testing of item membership via bitsets

pub struct EdgeList {
    /// store items per edge as bitset
    edges: Vec<BitSet>,
    /// store identifier per edge
    identifiers: Vec<usize>,
    /// store largest element per edge
    max_item: Vec<usize>,
}

type Edge = usize;

pub struct EdgeListIterator<'a> {
    edge_iterator: <&'a Vec<BitSet> as IntoIterator>::IntoIter,
    id_iterator: <&'a Vec<usize> as IntoIterator>::IntoIter,
}

pub struct SelectionIterator<'a> {
    query: ItemSeq<'a>,
    edge_list: &'a EdgeList,
    position: usize,
}

impl Link for EdgeList {
    type Edge = Edge;
    type SelectionIter<'a> = SelectionIterator<'a> where Self: 'a;
    
    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)> {
	let num_edges = self.edges.len();
	(0 .. num_edges)
	    .filter_map( |index| {
		match self.cut_off_subset( index, query ) {
		    Some( remainder ) => {
			let edge = self.identifiers[ index ];
			Some( (edge, remainder) )
		    },
		    None => None,
		}
	    }).collect()
    }

    fn light_select <'q> ( &'q self, query: ItemSeq<'q> ) -> Self::SelectionIter<'q> {
	let select_iter = SelectionIterator{ query, position: 0, edge_list: self };
	select_iter
    }

    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)> {
	assert!( !sequence.is_empty() );

	let index = self.search_edge_by_head( sequence[ 0 ]);
	if index.is_err() {
	    return None;
	}

	let index = index.expect( "tested success" );
	let label = &self.edges[ index ];
	let mut label_iter = label.iter();
	label_iter.next(); // checked the head already
	// let seq_iter = &sequence[ 1 ..].iter();

	// search for mismatching position
	let mut matching_items = 1;
	for (label_item, seq_item) in label_iter.zip( sequence[ 1 ..].iter()) {
	    if label_item != *seq_item {
		break;
	    }
	    matching_items += 1;
	}

	let edge = self.identifiers[ index ];
	Some( (edge, matching_items, matching_items == label.len()) )
    }

    fn add( &mut self, label: ItemSeq ) -> Self::Edge {
	assert!( !label.is_empty() );

	let edge_id = self.edges.len();
	let insert_position = self.search_edge_by_head( label[ 0 ]);
	// if there's an edge already, there's nothing to do
	if let Ok( index ) = insert_position {
	    return self.identifiers[ index ];
	}
	
	let max_item = label.last().expect( "non-empty by precondition" );
	let mut label_set = BitSet::with_capacity( *max_item );
	for item in label {
	    label_set.insert( *item  );
	}

	let insert_position = insert_position.expect_err( "no edge with head by precondition" );
	self.edges.insert( insert_position, label_set );
	self.identifiers.insert( insert_position, edge_id );
	self.max_item.insert( insert_position, *max_item );
	edge_id
    }

    fn split( &mut self, edge: &Self::Edge, position: usize ) -> ItemVec {
	let index = self.identifiers.iter().enumerate().find_map( |(index, ident)| if edge == ident { Some( index )} else { None })
	    .expect( "edge is valid identifier" );
	let label = self.edges.get_mut( index ).expect( "edge is valid by precondition" );
	let label_iter = label.iter();

	assert!( position > 0, "splitting shall never remove an edge completely" ); 
	let mut label_iter = label_iter.skip( position - 1 );
	let new_last = label_iter.next();
	if new_last.is_none() { // nothing is split
	    return Vec::new(); 
	}

	// split remaining elements of iterator
	self.max_item[ index ] = new_last.expect( "tested is some" );
	let drop_items: ItemVec = label_iter.collect();
	for item in &drop_items {
	    label.remove( *item );
	}
	drop_items
    }    
}

// TODO remove
/// Calculates remainder of query obtained by removing all elements from query that are in label.
/// Returns the remainder if all elements in query smaller than the greatest element in label are contained in label.
/// Pre: label is non-empty
fn cut_off_maximal_subsequence<'q> ( query: ItemSeq<'q>, label: &BitSet ) -> Option<ItemSeq<'q>> {
    // This takes most of the time! There does not seem to be a method to get the greatest element from a bitset/bitvec
    let label_max = label.iter().max().expect( "label is not empty" );
    let mut pos = 0;
    for query_item in query.iter().take_while( |item| **item <= label_max ) {
	if !label.contains( *query_item ) {
	    return None;
	}
	pos += 1;
    }
    
    return Some( &query[pos ..] )
}

impl Default for EdgeList {
    fn default() -> Self {
	EdgeList::new()
    }
}

impl <'a> IntoIterator for &'a EdgeList {
    type IntoIter = EdgeListIterator<'a>;
    type Item = (Edge, ItemVec);

    fn into_iter( self: &'a EdgeList ) -> EdgeListIterator<'a> {
	EdgeListIterator{
	    edge_iterator: self.edges.iter(),
	    id_iterator: self.identifiers.iter(),
	}
    }
}

impl EdgeList {

    pub fn new() -> EdgeList {
	EdgeList {
	    edges: Vec::new(),
	    identifiers: Vec::new(),
	    max_item: Vec::new(),
	}
    }

    pub fn get_id( &self, index: usize ) -> Edge {
	self.identifiers[ index ]
    }

    pub fn get_size( &self ) -> usize {
	self.edges.len()
    }

    /// Binary searches for the matching edge
    /// Returns a right edge if the search succeeds.
    /// Returns the index where to insert an edge with the given head if the search fails.
    fn search_edge_by_head( &self, head: Item ) -> Result<<Self as Link>::Edge, usize> {
	let compare = |label: &BitSet| label.iter().next().map_or( Ordering::Less, | label_head | label_head.cmp( &head ));
	self.edges.binary_search_by( compare )
    }

    /// Calculates remainder of query obtained by removing all elements from query that are in label.
    /// Returns the remainder if all elements in query smaller than the greatest element in label are contained in label.
    /// Pre: label is non-empty
    fn cut_off_subset <'q> ( &self, edge_index: usize, query: ItemSeq<'q> ) -> Option<ItemSeq<'q>> {
	if query.is_empty() { // empty set is always a subset
	    return Some( &[] )
	}

	let label = &self.edges[ edge_index ];
	let max = self.max_item[ edge_index ];

	let mut pos = 0;
	while pos < query.len() && query[ pos ] <= max {
	    if !label.contains( query[ pos ] ) {
		return None;
	    }
	    pos += 1;
	}

	if pos == query.len() { // all of query is contained
	    Some( &[] )
	} else { // exceeded the max at pos
	    let remainder = &query[ pos ..];
	    Some( remainder )
	}
    }
}

impl <'a> Iterator for EdgeListIterator<'a> {
    type Item = (Edge, ItemVec);

    fn next(&mut self) -> Option<Self::Item> {
	let next_edge = self.edge_iterator.next();
	let next_id = self.id_iterator.next();

	if let (Some( ident ), Some( label )) = (next_id, next_edge) {
	    Some( (*ident, label.iter().collect()) )
	} else {
	    None
	}
    }
}

impl <'a> Iterator for SelectionIterator<'a> {
    type Item = (Edge, ItemSeq<'a>);

    fn next( &mut self ) -> Option<Self::Item> {
	while self.position < self.edge_list.get_size() {
	    let index = self.position;
	    self.position += 1;
	    if let Some( remainder ) = self.edge_list.cut_off_subset( index, self.query ) {
		let edge_id = self.edge_list.get_id( index );
		return Some( (edge_id, remainder) );
	    }
	}
	None
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;
    use super::*;
    
    #[test]
    fn test_select() {
	let mut edges = EdgeList::new();
	let e1 = edges.add( &[1, 2, 3] );
	let e2 = edges.add( &[0, 2] );

	let selection = edges.select( &[0] );
	assert_eq!( selection.len(), 1 );
	assert_eq!( selection[ 0 ].0, e2 );
	assert_eq!( selection[ 0 ].1, &[] );

	let selection = edges.select( &[2, 3] );
	assert_eq!( selection.len(), 2 );
	let remainders: HashSet<ItemVec> = selection.iter().map( |edge_remainder| edge_remainder.1.iter().copied().collect() ).collect();
	assert!( remainders.contains( &vec!( 3 ) ));
	assert!( remainders.contains( &vec!()  ));

	let selection = edges.select( &[1, 3, 4] );
	assert_eq!( selection.len(), 1 );
	assert_eq!( selection[0].0, e1 );
	assert_eq!( selection[0].1, &[4] );

	// test query that does not always match the first element per branch
	let query = [3];
	let selection = edges.select( &query );
	assert_eq!( selection.len(), 2 );
	assert_eq!( selection[ 0 ], (e2, &query[ .. ]) );
	assert_eq!( selection[ 1 ], (e1, &query[ 1 .. ]) );
    }

    #[test]
    /// Test if the remainder of the query is determined correctly
    fn test_query_remaindering() {
	let mut edges = EdgeList::new();
	// do not insert in order, so edge order may be different
	let _ = edges.add( &[1, 2, 3] );
	let e1 = edges.add( &[0, 1, 2] );
	
	let query = [0, 1];
	let selection = edges.select( &query );
	assert_eq!( selection[ 0 ], (e1, &query[2 ..]) );

	let query = [0, 1, 3];
	let selection = edges.select( &query );
	assert_eq!( selection[ 0 ], (e1, &query[2 ..]) );
    }

    #[test]
    fn test_walk() {
	let mut edges = EdgeList::new();
	let e2 = edges.add( &[1, 2, 3] );
	let e1 = edges.add( &[0, 2, 3] );

	let result = edges.walk( &[1, 2, 3] );
	assert!( result.is_some() );
	assert_eq!( result.unwrap(), (e2, 3, true) );

	let result = edges.walk( &[2, 3] );
	assert!( result.is_none() );

	let result = edges.walk( &[0, 1] );
	assert!( result.is_some() );
	assert_eq!( result.unwrap(), (e1, 1, false) );
    }

    #[test]
    fn test_split() {
	let mut edges = EdgeList::new();
	let e2 = edges.add( &[2, 3, 4] );
	let e1 = edges.add( &[0, 1, 2, 3] );
	edges.split( &e1, 2 );

	let result = edges.walk( &[0, 1, 2] );
	assert_eq!( result.unwrap(), (e1, 2, true) );

	let result = edges.walk( &[2, 3, 4] );
	assert_eq!( result.unwrap(), (e2, 3, true) );
    }
}
