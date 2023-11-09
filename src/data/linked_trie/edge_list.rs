

use super::*;

pub struct EdgeList {
    // mapping from distinct starting item to complete label
    edges: HashMap<Item, ItemVec>,
}

pub struct HashMapIterAdaptor<'a> {
    iter: collections::hash_map::Iter<'a, Item, ItemVec>,
}

impl Link for EdgeList {
    type Edge = Item;

    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)> {
	self.edges.iter()
	    .filter_map( |(edge, label)| {
		let (next_pos, is_super) = is_partial_superset( query, &label );
		let remainder = &query[ next_pos .. ];
		if is_super { Some( (*edge, remainder) )} else { None }
	    }).collect()
    }

    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)> {
	assert!( !sequence.is_empty() );
	let head = sequence[ 0 ];
	if let Some( label ) = self.edges.get( &head ) {
	    let (next_pos, success) = is_prefix( sequence, &label );
	    Some( (head, next_pos, success) )
	} else {
	    None
	}
    }

    fn add( &mut self, label: ItemSeq ) -> Self::Edge {
	assert!( !label.is_empty() );
	let head = label[ 0 ];
	let last = *label.last().expect( "label is non-empty" );
	assert!(  !self.edges.contains_key( &head ));
	let mut label_set = BitSet::with_capacity( last );
	for item in label {
	    label_set.insert( *item );
	}
	let label: ItemVec = label.iter().map( |itm| *itm ).collect();
	self.edges.insert( head, label );
	head
    }

    fn split( &mut self, edge: &Self::Edge, position: usize ) -> ItemVec {
	let label = self.edges.get_mut( &edge ).expect( "pre: edge is valid" );
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
	EdgeList { edges: HashMap::new() }
    }
}

impl <'a> IntoIterator for &'a EdgeList {
    type Item = (<EdgeList as Link>::Edge, ItemVec);
    type IntoIter = HashMapIterAdaptor<'a>;

    fn into_iter(self) -> Self::IntoIter {
	HashMapIterAdaptor{ iter: self.edges.iter() }
    }
}

impl <'a> Iterator for HashMapIterAdaptor<'a> {
    type Item = (<EdgeList as Link>::Edge, ItemVec);

    fn next(&mut self) -> Option<Self::Item> {
	self.iter.next().map( |(edge, label)| (edge.clone(), label.clone()) )
    }
}
