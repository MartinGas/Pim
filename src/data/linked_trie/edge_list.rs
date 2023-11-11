
use rustc_hash::FxHashMap;

use super::*;

pub struct EdgeList {
    /// Stores the edges
    edges: Vec<ItemVec>,
    /// Maps starting item to edge index
    // mapping from distinct starting item to complete label
    head_to_index: FxHashMap<Item, usize>,
}

pub struct HashMapIterAdaptor<'a> {
    iter: <&'a Vec<ItemVec> as IntoIterator>::IntoIter,
    counter: usize,
}

impl Link for EdgeList {
    type Edge = usize;

    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)> {
	self.edges.iter().enumerate()
	    .filter_map( |(index, label)| {
		let (next_pos, is_super) = is_partial_superset( query, &label );
		let remainder = &query[ next_pos .. ];
		if is_super { Some( (index, remainder) )} else { None }
	    }).collect()
    }

    fn light_select <'q, 'e: 'q> ( &'e self, query: ItemSeq<'q> ) -> Box<dyn Iterator<Item = (Self::Edge, ItemSeq<'q>)> + 'q> {
	let iter = self.edges.iter().enumerate()
	    .filter_map( |(index, label)| {
		let (next_pos, is_super) = is_partial_superset( query, &label );
		let remainder = &query[ next_pos .. ];
		if is_super { Some( (index, remainder) )} else { None }
	    });
	Box::new( iter )
    }

    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)> {
	assert!( !sequence.is_empty() );
	let head = sequence[ 0 ];

	match self.head_to_index.get( &head ) {
	    Some( index ) => {
		let label = &self.edges[ *index ];
		let (next_pos, success) = is_prefix( sequence, &label );
		Some( (*index, next_pos, success) )
	    },
	    None => None,
	}
    }

    fn add( &mut self, label: ItemSeq ) -> Self::Edge {
	assert!( !label.is_empty() );

	let new_index = self.edges.len();
	let head = label[ 0 ];
	assert!( !self.head_to_index.contains_key( &head ));
	
	let mut label: ItemVec = label.iter().copied().collect();
	label.sort();
	label.dedup();

	self.edges.push( label );
	self.head_to_index.insert( head, new_index);

	new_index
    }

    fn split( &mut self, edge: &Self::Edge, position: usize ) -> ItemVec {
	let label = self.edges.get_mut( *edge ).expect( "pre: edge is valid" );
	label.split_off( position )
    }
}

/// Returns next position in items after last match and indicates if the path may be a superset of items
fn is_partial_superset( items: ItemSeq, path: &ItemVec ) -> (usize, bool) {
    let mut path_iter = path.iter();
    let mut next_position = 0;

    // check whether query items are on the path
    for query_item in items.iter() {
	let mut path_item = path_iter.next();
	while path_item.map_or( false, |item| item < query_item ) {
	    path_item = path_iter.next();
	}

	// all items on path are less than the current item
	if path_item.is_none() {
	    return (next_position, true);
	}
	let path_item = path_item.unwrap();
	// next item on path is larger, so query item is not on path
	if query_item < path_item {
	    return (next_position, false);
	}

	assert!( query_item == path_item, "proceed if equal" );
	next_position += 1;
    }
    (next_position, true)
}

/// returns next position in items after last match and indicates if the path is a prefix of items.
fn is_prefix( items: ItemSeq, path: &ItemVec ) -> (usize, bool) {
    let mut query_iter = items.iter();
    let mut path_iter = path.iter();

    let mut query_item = query_iter.next();
    let mut path_item = path_iter.next();
    let mut position = 0;
    while query_item.is_some() && path_item.is_some() {
	if query_item != path_item {
	    return (position, false);
	}

	query_item = query_iter.next();
	path_item = path_iter.next();
	position += 1;
    }

    // all query items matched
    // path is prefix if path iterator is exhausted
    return (position, path_item.is_none())
}

impl Default for EdgeList {
    fn default() -> Self {
	EdgeList { edges: Vec::new(), head_to_index: FxHashMap::default() }
    }
}

impl <'a> IntoIterator for &'a EdgeList {
    type Item = (<EdgeList as Link>::Edge, ItemVec);
    type IntoIter = HashMapIterAdaptor<'a>;

    fn into_iter(self) -> Self::IntoIter {
	HashMapIterAdaptor{ iter: self.edges.iter(), counter: 0 }
    }
}

impl <'a> Iterator for HashMapIterAdaptor<'a> {
    type Item = (<EdgeList as Link>::Edge, ItemVec);

    fn next(&mut self) -> Option<Self::Item> {
	let edge = self.counter;
	self.counter += 1;
	self.iter.next().map( |label| (edge, label.clone()) )
    }
}
