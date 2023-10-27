

use super::*;

pub struct EdgeList {
    // mapping from distinct starting item to complete label
    edges: HashMap<Item, ItemVec>,
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

impl Default for EdgeList {
    fn default() -> Self {
	EdgeList { edges: HashMap::new() }
    }
}

impl <'a> IntoIterator for &'a EdgeList {
    type Item = (&'a Item, &'a ItemVec);
    type IntoIter = collections::hash_map::Iter<'a, Item, ItemVec>;

    fn into_iter(self) -> Self::IntoIter {
	self.edges.iter()
    }
}
