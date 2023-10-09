
use std::hash::Hash;
use std::collections::{self, HashMap};
use std::default::Default;

use bit_set::BitSet;

use super::{Count, Item};

/// Trie with natural ordering of Items
pub struct Trie {
    root: Node<EdgeList, <EdgeList as Link>::Edge>,
}

/// stores edges to children
struct Node<L, E> {
    edges: L,
    /// invariant: node order is the same as corresponding edges in the edge matrix
    children: HashMap<E, Node<L, E>>,
    support: Count,
}

pub struct NodeIterator<N, Eit> {
    // Stores sequence of node-iterator pairs. The end of the vector is the front of the stack.
    visit_stack: Vec<(N, Eit)>,
}

/// A sorted sequence of items.
type ItemSeq<'a> = &'a [Item];
/// A sorted vector of items
type ItemVec = Vec<Item>;

/// Collection of edges labeled by ordered item sets.
trait Link {
    type Edge; // representation of an edge

    /// returns the edge that matches the query and the remainder of the query
    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)>;

    /// Returns the edge, size of the prefix and whether the prefix matches completely, if there is a prefix.
    /// Pre: sequence is not empty
    /// Pre: there is an edge for the first element of sequence
    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)>;

    /// Adds an edge to the collection.
    /// Pre: label is not empty
    /// Pre: there is no partially matching edge for the given label.
    /// Returns the created edge
    fn add( &mut self, label: ItemSeq ) -> Self::Edge;

    /// Splits the edge from the given position onwards and returns the label split off
    /// Pre: edge refers to a valid edge
    fn split( &mut self, edge: Self::Edge, position: usize ) -> ItemVec;
}

struct EdgeList {
    // mapping from distinct starting item to complete label
    edges: HashMap<Item, ItemVec>,
}

impl Trie {

    pub fn new() -> Trie {
	Trie{ root: Node::new( 0 ) }
    }

    // used for debugging
    #[allow(dead_code)]
    pub fn print( &self ) {
	let mut buf = Vec::new();
	self.root.print( &mut buf )
    }

    /// Returns the support count of the query.
    pub fn query_support( &self , mut query: Vec<Item> ) -> Count {
	query.sort();
	self.root.query_support( &query )
    }

    /// Adds transaction t to the trie, increasing all supports
    pub fn add( &mut self, mut transaction: Vec<Item> ) {
	transaction.sort();
	self.root.add( &transaction )
    }

    pub fn iterate <'a> ( &'a self ) -> TrieIterator<'a> {
	NodeIterator::new( &self.root )
    }
}

type TrieIterator<'a> = NodeIterator<&'a Node<EdgeList, <EdgeList as Link>::Edge>,
				     <&'a EdgeList as IntoIterator>::IntoIter>;

impl <L, E> Node<L, E> {
    pub fn get_support( &self ) -> Count { self.support }
    pub fn get_edges( &self ) -> &L { &self.edges }
}

impl <L: Default, E> Node<L, E> {

    pub fn new( support: Count ) -> Node<L, E> {
	Node{
	    support,
	    edges: Default::default(),
	    children: HashMap::new(),
	}
    }
}

impl <L: Link> Node<L, L::Edge> where L::Edge: Eq + Hash {

    pub fn get_child( &self, edge: &L::Edge ) -> Option<&Node<L, L::Edge>> {
	self.children.get( edge )
    }

    /// visits all matching nodes for the sorted query from position onwards
    /// return the sum of the supports of terminals
    pub fn query_support( &self, query: ItemSeq ) -> Count {
	// solved query if all items are accounted for
	if query.is_empty() {
	    return self.support;
	}

	let selection = self.edges.select( &query );
	let mut sum: Count = 0;
	for (edge, remainder) in selection {
	    assert!( self.children.contains_key( &edge ));
	    let child = &self.children[ &edge ];
	    sum += child.query_support( remainder );
	}
	sum
    }

    /// Pushes a prefix down to the child
    /// Pre: there is no edge for the given edge_label
    fn push( &mut self, edge_label: ItemSeq, child: Node<L, L::Edge> ) {
	// todo: better interface, this is overblown
	let edge = self.edges.add( edge_label );
	assert!( !self.children.contains_key( &edge ) ); // there is a slot for every edge
	self.children.insert( edge, child );
    }

    /// Prints the items at this node and the support
    /// Used for debugging.
    #[allow(dead_code)]
    pub fn print <'a> ( &'a self, items: &mut ItemVec ) where
	&'a L: IntoIterator<Item = (&'a L::Edge, &'a ItemVec)> {
	println!( "{} : {}", format_items( items.iter() ), self.support );

	let edges: &L = &self.edges;
	for (edge, label) in edges.into_iter() {
	    assert!( self.children.contains_key( edge ));
	    for item in label {
		items.push( *item );
	    }
	    self.children[ edge ].print( items );
	    for _ in label {
		items.pop();
	    }
	}
    }
}

impl <L: Default + Link> Node<L, L::Edge> where L::Edge: Eq + Hash {
    /// Extends an existing branch or creates a new one
    pub fn add( &mut self, transaction: ItemSeq ) {
	self.support += 1;
	if transaction.is_empty() {
	    return;
	}

	if let Some( (edge, add_pos, is_complete) ) = self.edges.walk( transaction ) {
	    assert!( self.children.contains_key( &edge ) );
	    let child = self.children.get_mut( &edge ).expect( "there is an edge to a valid node" );
	    // there is an edge
	    if !is_complete {
		let excess = self.edges.split( edge, add_pos );
		child.push( &excess, Node::new( child.support ));
	    }
	    // if add_pos is out of bounds we simply pass on the empty slice
	    let remainder = transaction.get( add_pos .. ).unwrap_or( &[] );
	    child.add( remainder );
	} else {
	    // we need a new edge and node
	    let edge = self.edges.add( &transaction );
	    assert!( !self.children.contains_key( &edge ) );
	    self.children.insert( edge, Node::new( 1 ));
	}
    }
}

impl <'a, L, E> NodeIterator<&'a Node<L, E>, <&'a L as IntoIterator>::IntoIter> where &'a L: IntoIterator {

    pub fn new( start: &'a Node<L, E> ) -> NodeIterator<&'a Node<L, E>, <&'a L as IntoIterator>::IntoIter> {
	let edge_iterator = start.get_edges();
	NodeIterator {
	    visit_stack: vec!( (start, edge_iterator.into_iter()) ),
	}
    }
}

// implementation of the node iterator for edge lists (references thereof)
impl <'a> NodeIterator<&'a Node<EdgeList, <EdgeList as Link>::Edge>, <&'a EdgeList as IntoIterator>::IntoIter> {
    
}

impl <'a> Iterator for NodeIterator<&'a Node<EdgeList, <EdgeList as Link>::Edge>, <&'a EdgeList as IntoIterator>::IntoIter> {

    type Item = (&'a ItemVec, Count);

    fn next(&mut self) -> Option<Self::Item> {
	if self.visit_stack.is_empty() { // nothing more to see
	    return None;
	}

	let (node, edge_iterator) = self.visit_stack.last_mut().expect( "returned if stack was empty" );

	// use the next down-edge if possible
	if let Some( (edge, label) ) = edge_iterator.next() {
	    let child = node.get_child( edge ).expect( "every edge connects to a child" );
	    // get the info needed
	    let count = child.get_support();
	    // push next level onto the stack
	    let child_edge_iterator = child.get_edges().into_iter();
	    self.visit_stack.push( (child, child_edge_iterator) );
	    return Some( (label, count) );
	}

	// no down-edge, go up and try again
	self.visit_stack.pop();
	self.next()
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

    fn split( &mut self, edge: Self::Edge, position: usize ) -> ItemVec {
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

// todo move it somewhere where it's useful
#[allow(dead_code)]
fn format_items <'a, I> ( items: I ) -> String where I: Iterator<Item = &'a Item> {
    let mut string = String::new();
    for item in items {
	string.push_str( format!( "{} ", item ).as_str() );
    }
    string
}

#[cfg(test)]
mod test {
    use super::*;

    fn from_vec( elements: Vec<Item> ) -> BitSet {
	BitSet::from_iter( elements )
    }

    pub fn build_trie_from_complete_data( data: &Vec<Vec<Item>> ) -> Trie {
	let mut trie = Trie::new();
	for transaction in data {
	    trie.add( transaction.clone() );
	}
	trie
    }
    
    #[test]
    fn add_and_query_support() {
	let data = vec!(
	    vec!( 1, 2, 3 ),
	    vec!( 0, 2, 1 ),
	    vec!( 0, 1, 2 ),
	    vec!( 0, 2, 4 ),
	    vec!( 4, 3, 1 )
	);

	let trie = build_trie_from_complete_data( &data );
	let expectations = vec!(
	    (vec!( 0 ), 3),
	    (vec!( 0, 1 ), 2),
	    (vec!( 3, 4 ), 1),
	    (vec!( 2, 4 ), 1),
	    (vec!( 0, 3 ), 0),
	    (vec!( 1, 5 ), 0), // ask for something that's not even there
	);

	trie.print();
	
	for (items, expected) in expectations {
	    let calculated = trie.query_support( items.clone() );
	    assert_eq!( calculated, expected, "{}", format_items( items.iter() ) );
	}
    }

    /// Compresses some sequences into a trie. Visits all sequences afterwards.
    #[test]
    fn test_iteration() {
	let data = vec!(
	    vec!( 0, 1, 2, 3, 4 ),
	    vec!( 2, 3, 4 ),
	    vec!( 0, 3, 4 ),
	    vec!( 0, 1, 2 ),
	    vec!( 2, 3, 4 ), // duplicate
	    vec!( 2, ),
	);
	let trie = build_trie_from_complete_data( &data );

	// terminal nodes visited in order
	let expectations = vec!(
	    (vec!( 0 ), 3),
	    (vec!( 1, 2 ), 2),
	    (vec!( 3, 4 ), 1),
	    (vec!( 3, 4 ), 1),
	    (vec!( 2 ), 3),
	    (vec!( 3, 4 ), 2),
	);

	trie.print();
	println!();

	let zipper = trie.iterate().zip( expectations.iter() );
	for ((expected_chunk, expected_count), (real_chunk, real_count)) in zipper {
	    assert_eq!( expected_chunk, real_chunk );
	    assert_eq!( expected_count, *real_count );
	}
    }
}
