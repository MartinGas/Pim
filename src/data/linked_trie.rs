
mod edge_list;
mod edge_list2;
mod skip_graph;

use std::hash::Hash;
use std::collections::{self, HashMap};
use std::default::Default;
use std::ops::DerefMut;

use bit_set::BitSet;

use super::{Count, Item};

pub trait TrieInterface {
    type Iterator<'a> where Self: 'a;

    /// Returns the support count of the subset query.
    fn query_subset_support( &self , query: Vec<Item> ) -> Count;

    fn query_prefix_support( &self, query: Vec<Item> ) -> Count;

    /// Adds transaction t to the trie, increasing all supports by count
    fn add( &mut self, transaction: Vec<Item>, count: Count );

    /// Iterates over the nodes and accumulate state with a comsumer
    fn iterate_and_consume <'a, C> ( &'a self, consumer: C ) -> Self::Iterator<'a> where C: Consumer<ItemVec> + 'static;
}

/// Trie with natural ordering of Items
pub struct Trie<L: Link> {
    root: Node<L, L::Edge>,
    node_builder: Box<dyn NodeBuilder<L, L::Edge>>,
}

/// Trie variants for external use
pub type EdgeListTrie = Trie<edge_list::EdgeList>;
pub type EdgeListTrieBetter = Trie<edge_list2::EdgeList>;
pub type SkipGraphTrie = Trie<skip_graph::SkipGraph>;

/// wraps an interator to hide the internal generic parameters
pub struct TrieIterator<'a, L: Link + 'a> where &'a L: IntoIterator {
    iterator: NodeIterator<&'a Node<L, L::Edge>,
			   <&'a L as IntoIterator>::IntoIter,
			   Box<dyn Consumer<ItemVec>>>,
}

/// Iterator variants for external use
pub type EdgeListTrieIterator<'a> = TrieIterator<'a, edge_list::EdgeList>;

/// Types that consume sequences fed in chunks to them during the traversal.
pub trait Consumer<L> {
    /// Notifies the consumer that the traversal entered a new node via an edge with the given label
    fn enter( &mut self, label: L );
    /// Notifies the consumer that the traversal leaves the current node
    fn leave( &mut self );
}

trait NodeBuilder<L, E> {
    fn build( &self, support: Count ) -> Node<L, E>;
}

struct DefaultNodeBuilder;
/// stores edges to children
struct Node<L, E> {
    edges: L,
    /// invariant: node order is the same as corresponding edges in the edge matrix
    children: HashMap<E, Node<L, E>>,
    support: Count,
}

struct NodeIterator<N, Eit, C> {
    // Stores sequence of node-iterator pairs. The end of the vector is the front of the stack.
    visit_stack: Vec<(N, Eit)>,
    consumer: Option<C>,
}

/// To keep consumer-related code the same we use them as trait objects through boxes
impl <L> Consumer<L> for Box<dyn Consumer<L>> {
    fn enter( &mut self, label: L ) { self.as_mut().enter( label ) }
    fn leave( &mut self ) { self.as_mut().leave() }
}

/// A sorted sequence of items.
type ItemSeq<'a> = &'a [Item];
/// A sorted vector of items
type ItemVec = Vec<Item>;

/// Collection of edges labeled by ordered item sets.
pub trait Link {
    type Edge; // representation of an edge

    /// returns the edge that matches the subset query and the remainder of the query
    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)>;

    /// Determines if there is an edge that is a partial prefix of the sequence.
    /// Returns the edge, size of the prefix and whether the complete prefix matches the sequence,
    /// if there is a prefix.
    /// Pre: sequence is not empty
    /// Pre: there is an edge for the first element of sequence
    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)>;

    /// Adds an edge to the collection.
    /// Pre: label is not empty
    /// Pre: there is no partially matching edge for the given label.
    /// Returns the created edge
    fn add( &mut self, label: ItemSeq ) -> Self::Edge;

    /// Splits the edge from the given index onwards and returns the label split off
    /// Pre: edge refers to a valid edge
    fn split( &mut self, edge: &Self::Edge, position: usize ) -> ItemVec;
}

impl Trie<edge_list::EdgeList> {
    pub fn new_with_edgelist() -> Trie<edge_list::EdgeList> {
	let node_builder = DefaultNodeBuilder;
	let root = node_builder.build( 0 );
	Trie{ root,
	      node_builder: Box::new( node_builder ),
	}
    }    
}

impl Trie<edge_list2::EdgeList> {
    pub fn new_with_edgelist_better() -> Trie<edge_list2::EdgeList> {
	let node_builder = DefaultNodeBuilder;
	let root = node_builder.build( 0 );
	Trie{ root,
	      node_builder: Box::new( node_builder ),
	}
    }
}

impl Trie<skip_graph::SkipGraph> {
    pub fn new_with_skipgraph( number_of_items: usize ) -> SkipGraphTrie {
	// simplify: every skip graph has all items as nodes
	let node_builder = move |support| Node::new( support, skip_graph::SkipGraph::new( number_of_items ));
	let root = node_builder.build( 0 );
	Trie{ root,
	      node_builder: Box::new( node_builder ),
	}
    }
}

impl <L: Link> TrieInterface for Trie<L> where
    L::Edge: Eq + Hash + Clone,
    for<'a> &'a L: IntoIterator<Item = (L::Edge, ItemVec)>,
{
    type Iterator<'a> = TrieIterator<'a, L> where L: 'a;
    
    /// Returns the support count of the subset query.
    fn query_subset_support( &self, query: Vec<Item> ) -> Count {
	let mut query = query;
	query.sort();
	self.root.query_subset_support( &query )
    }

    fn query_prefix_support( &self, query: Vec<Item> ) -> Count {
	let mut query = query;
	query.sort();
	self.root.query_prefix_support( &query )
    }

    /// Adds transaction t to the trie, increasing all supports by count
    fn add( &mut self, mut transaction: Vec<Item>, count: Count ) {
	transaction.sort();
	self.root.add( &transaction, count, self.node_builder.as_ref() );
    }

    fn iterate_and_consume <'a, C> ( &'a self, consumer: C ) -> Self::Iterator<'a> where C: Consumer<ItemVec> + 'static {
	TrieIterator::new_with_consumer( &self.root, consumer )
    }
    
}

impl <'a, L: Link + 'a> Trie<L> where
    for <'b> &'b L: IntoIterator<Item = (L::Edge, ItemVec)>,
    L::Edge: Eq + Hash + Clone
{

    // used for debugging
    #[allow(dead_code)]
    pub fn print( &'a self ) {
	let mut buf = Vec::new();
	self.root.print( &mut buf )
    }

    #[allow(dead_code)]
    pub fn iterate( &'a self ) -> TrieIterator<'a, L> {
	TrieIterator::new( &self.root )
    }

    pub fn iterate_with_consumer <C> ( &'a self, consumer: C ) -> TrieIterator<'a, L> where
	C: Consumer<ItemVec> + 'static,
    {
	TrieIterator::new_with_consumer( &self.root, consumer )
    }
}

// impl <'a, L: Link + 'a> IntoIterator for &'a Trie<L> where
//     &'a L: IntoIterator<Item = (L::Edge, ItemVec)>,
// {
//     type IntoIter = TrieIterator<'a, L>;

//     fn into_iter(self) -> Self::IntoIter {
// 	TrieIterator::new( &self.root )
//     }
// }

impl <'a, L: Link> TrieIterator<'a, L> where &'a L: IntoIterator + 'a {
    #[allow(dead_code)]
    fn new( start: &'a Node<L, L::Edge> ) -> TrieIterator<'a, L> {
	TrieIterator{ iterator: NodeIterator::new( start, None ) }
    }

    fn new_with_consumer <C> ( start: &'a Node<L, L::Edge>, consumer: C ) -> TrieIterator<'a, L> where
	C: Consumer<ItemVec> + 'static
    {
	let consumer = Box::new( consumer );
	TrieIterator{ iterator: NodeIterator::new( start, Some( consumer )) }
    }
}

impl <'a, L: Link + 'a> Iterator for TrieIterator<'a, L> where
    &'a L: IntoIterator<Item = (L::Edge, ItemVec)>,
    L::Edge: Eq + Hash + Clone,
{
    type Item = (ItemVec, Count);

    fn next( &mut self ) -> Option<Self::Item> {
	self.iterator.next()
    }
}

impl <L: Default, E> NodeBuilder<L, E> for DefaultNodeBuilder {
    fn build( &self, support: Count ) -> Node<L, E> {
	Node::new( support, L::default() )
    }
}

impl <F, L, E> NodeBuilder<L, E> for F where F: Fn(Count) -> Node<L, E> {
    fn build( &self, support: Count ) -> Node<L, E> {
	self( support )
    }
}

impl <L, E> Node<L, E> {
    pub fn get_support( &self ) -> Count { self.support }
    pub fn get_edges( &self ) -> &L { &self.edges }
}

impl <L, E> Node<L, E> {

    pub fn new( support: Count, edges: L ) -> Node<L, E> {
	Node{
	    support,
	    edges,
	    children: HashMap::new(),
	}
    }
}

impl <L: Link> Node<L, L::Edge> where L::Edge: Eq + Hash + Clone {

    pub fn get_child( &self, edge: &L::Edge ) -> Option<&Node<L, L::Edge>> {
	self.children.get( edge )
    }

    /// Visits all nodes that represent supersets of query.
    /// Returns the sum of the supports of terminals.
    pub fn query_subset_support( &self, query: ItemSeq ) -> Count {
	// solved query if all items are accounted for
	if query.is_empty() {
	    return self.support;
	}

	let selection = self.edges.select( &query );
	let mut sum: Count = 0;
	for (edge, remainder) in selection {
	    assert!( self.children.contains_key( &edge ));
	    let child = &self.children[ &edge ];
	    sum += child.query_subset_support( remainder );
	}
	sum
    }

    /// Visits the nodes on the prefix path.
    /// Returns the support at the terminal.
    pub fn query_prefix_support( &self, sequence: ItemSeq ) -> Count {
	// reached terminal
	if sequence.is_empty() {
	    return self.support;
	}

	if let Some( (edge, pos, is_match) ) = self.edges.walk( sequence ) {
	    if !is_match { // there is a gap in the sequence
		return 0;
	    }

	    // found an edge that is a complete prefix
	    let child = &self.children[ &edge ];
	    child.query_prefix_support( &sequence[ pos ..] )
	} else { // there is no matching edge
	    0
	}
    }

        /// Extends an existing branch or creates a new one
    pub fn add( &mut self, transaction: ItemSeq, count: Count, node_builder: & dyn NodeBuilder<L, L::Edge> ) {
	self.support += count;
	if transaction.is_empty() {
	    return;
	}

	if let Some( (edge, add_pos, is_complete) ) = self.edges.walk( transaction ) {
	    assert!( self.children.contains_key( &edge ) );
	    // if add_pos is out of bounds we simply pass on the empty slice
	    let remainder = transaction.get( add_pos .. ).unwrap_or( &[] );

	    // there is an edge
	    if !is_complete {
		let new_intermediate = node_builder.build( 0 );
		self.split( new_intermediate, edge.clone(), add_pos );
		self.children.get_mut( &edge ).expect( "replaced child" ).add( remainder, count, node_builder );
	    } else {
		let child = self.children.get_mut( &edge ).expect( "Edge leads to child" );
		child.add( remainder, count, node_builder );
	    }
	} else {
	    // we need a new edge and node
	    let edge = self.edges.add( &transaction );
	    assert!( !self.children.contains_key( &edge ) );
	    self.children.insert( edge, node_builder.build( count ));
	}
    }

    /// Handles incomplete match where part of an edge needs to be split off while adding.
    /// Replace the current child by new_intermediate and push down the child.
    fn split( &mut self, new_intermediate: Node<L, L::Edge>, edge: L::Edge, split_pos: usize ) {
	// Only part of the prefix matched, so we need to introduce an new node to branch in between.
	let excess = self.edges.split( &edge, split_pos );
	assert!( !excess.is_empty() );
	
	// push the current child down to intermediate
	let pushed_down_child = self.children.insert( edge.clone(), new_intermediate ).expect( "Push down existing child" );
	let new_intermediate = self.children.get_mut( &edge ).expect( "Inserted intermediate child" );
	new_intermediate.support = pushed_down_child.support;
	new_intermediate.push( &excess, pushed_down_child );
    }

    /// Pushes a prefix down to the child
    /// Pre: there is no edge for the given edge_label
    fn push( &mut self, edge_label: ItemSeq, child: Node<L, L::Edge> ) {
	let edge = self.edges.add( edge_label );
	assert!( !self.children.contains_key( &edge ) ); // there is a slot for every edge
	self.children.insert( edge, child );
    }

    /// Prints the items at this node and the support
    /// Used for debugging.
    #[allow(dead_code)]
    pub fn print <'a> ( &'a self, items: &mut ItemVec ) where
	&'a L: IntoIterator<Item = (L::Edge, ItemVec)> {
	println!( "{} : {}", format_items( items.iter() ), self.support );

	let edges: &L = &self.edges;
	for (edge, label) in edges.into_iter() {
	    assert!( self.children.contains_key( &edge ));
	    for item in &label {
		items.push( *item );
	    }
	    self.children[ &edge ].print( items );
	    for _ in label {
		items.pop();
	    }
	}
    }
}

impl <L: Link> Node<L, L::Edge> where L::Edge: Eq + Hash + Clone {

}

impl <'a, L, E, C> NodeIterator<&'a Node<L, E>, <&'a L as IntoIterator>::IntoIter, C> where &'a L: IntoIterator {

    pub fn new( start: &'a Node<L, E>, consumer: Option<C> ) -> NodeIterator<&'a Node<L, E>, <&'a L as IntoIterator>::IntoIter, C> {
	let edge_iterator = start.get_edges();
	NodeIterator {
	    visit_stack: vec!( (start, edge_iterator.into_iter()) ),
	    consumer,
	}
    }
}

// lifetimes:
// 'eg is borrow of edges for the purpose of iteration
// 'a is a temporary borrow for the time of the method only
impl <'eg, 'a, L: Link, C> NodeIterator<&'eg Node<L, L::Edge>, <&'eg L as IntoIterator>::IntoIter, C> where
    &'eg L: IntoIterator,
    C: DerefMut<Target = dyn Consumer<ItemVec>>,
{
    fn enter( &mut self, child: &'eg Node<L, L::Edge>, label: ItemVec ) {
	if let Some( consumer ) = &mut self.consumer {
	    consumer.enter( label );
	}
	// push next level onto the stack
	let child_edge_iterator = child.get_edges().into_iter();
	self.visit_stack.push( (child, child_edge_iterator) );
    }

    fn leave( &mut self ) {
	if let Some( consumer ) = &mut self.consumer {
	    consumer.leave();
	}
	self.visit_stack.pop();
    }
}

impl <'eg, 'a, L: Link, C> Iterator for NodeIterator<&'eg Node<L, L::Edge>, <&'eg L as IntoIterator>::IntoIter, C> where
    &'eg L: IntoIterator<Item = (L::Edge, ItemVec)>,
    C: DerefMut<Target = dyn Consumer<ItemVec>>,
    L::Edge: Eq + Hash + Clone,
{

    type Item = (ItemVec, Count);

    fn next(&mut self) -> Option<Self::Item> {
	if self.visit_stack.is_empty() { // nothing more to see
	    return None;
	}

	let (node, edge_iterator) = self.visit_stack.last_mut().expect( "returned if stack was empty" );

	// use the next down-edge if possible
	if let Some( (edge, label) ) = edge_iterator.next() {
	    let child = node.get_child( &edge ).expect( "every edge connects to a child" );
	    // get the info needed
	    let count = child.get_support();
	    // push
	    self.enter( child, label.clone() );
	    return Some( (label, count) );
	}

	// no down-edge, go up and try again
	self.leave();
	self.next()
    }
}

impl Link for skip_graph::SkipGraph {
    // edges are represented by a sequence id
    type Edge = usize;

    fn select <'e, 'q> ( &'e self, query: ItemSeq<'q> ) -> Vec<(Self::Edge, ItemSeq<'q>)> {
	let subsequence_to_visits = self.get_sequence_with_subsequence( &query );
	let mut edge_and_remainder = Vec::with_capacity( subsequence_to_visits.len() );
	for (edge, visits) in subsequence_to_visits {
	    let remainder = &query[ visits .. ]; // unvisited 
	    // let remainder = if visits < query.len() { &query[ visits + 1 ..] } else { &[] };
	    edge_and_remainder.push( (edge, remainder) );
	}
	edge_and_remainder
    }

    fn walk( &self, sequence: ItemSeq ) -> Option<(Self::Edge, usize, bool)> {
	let prefix_to_visits = self.get_sequence_with_prefix( &sequence );
	// there is only one edge for every starting letter
	assert!( prefix_to_visits.len() < 2 );
	
	for (edge, (visits, is_complete)) in prefix_to_visits {
	    return Some( (edge, visits, is_complete) );
	}
	None
    }

    fn add( &mut self, label: ItemSeq ) -> Self::Edge {
	assert!( !label.is_empty() ); // handle empty labels
	self.add( label )
    }

    fn split( &mut self, edge: &Self::Edge, position: usize ) -> ItemVec {
	assert!( position > 0 ); // skip graph cannot delete sequences (yet)
	self.truncate( *edge, position - 1 )
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

    pub fn build_trie_from_complete_data( data: &Vec<Vec<Item>> ) -> EdgeListTrie {
	let mut trie = Trie::new_with_edgelist();
	for transaction in data {
	    trie.add( transaction.clone(), 1 );
	}
	trie
    }

    fn add_and_query_subset_support <T: TrieInterface > ( mut trie: T ) {
	// number of items: 6 (last query asks for item 5)
	let data = vec!(
	    vec!( 1, 2, 3 ),
	    vec!( 0, 2, 1 ),
	    vec!( 0, 1, 2 ),
	    vec!( 0, 2, 4 ),
	    vec!( 4, 3, 1 )
	);
	for transaction in data {
	    trie.add( transaction, 1 );
	}

	let expectations = vec!(
	    (vec!( 0 ), 3),
	    (vec!( 0, 1 ), 2),
	    (vec!( 3, 4 ), 1),
	    (vec!( 2, 4 ), 1),
	    (vec!( 0, 3 ), 0),
	    (vec!( 1, 5 ), 0), // ask for something that's not even there
	);
	
	for (items, expected) in expectations {
	    let calculated = trie.query_subset_support( items.clone() );
	    assert_eq!( calculated, expected, "{}", format_items( items.iter() ) );
	}
    }
    
    #[test]
    fn test_edgelist_add_and_query_subset_support() {
	let trie = Trie::new_with_edgelist();
	add_and_query_subset_support( trie );
    }

    #[test]
    fn test_better_edgelist_add_and_query_subset_support() {
	let trie = Trie::new_with_edgelist_better();
	add_and_query_subset_support( trie );
    }

    #[test]
    fn test_skipgraph_add_and_query_subset_support() {
	let trie = Trie::new_with_skipgraph( 6 );
	add_and_query_subset_support( trie );
    }

    fn add_and_query_prefix_support <T: TrieInterface> ( mut trie: T ) {
	// number of items: 3
	let data = vec!(
	    vec!( 0 ),
	    vec!( 0, 1 ),
	    vec!( 0, 2 ),
	    vec!( 2 )
	);
	for transaction in data {
	    trie.add( transaction, 1 );
	}

	let expectations = vec!(
	    (vec!( 0 ), 3),
	    (vec!( 0, 1 ), 1),
	    (vec!( 2 ), 1 ),
	    (vec!( 1, 2 ), 0),
	);

	for (prefix, expected) in expectations {
	    let calculated = trie.query_prefix_support( prefix.clone() );
	    assert_eq!( calculated, expected, "{prefix:?}" );
	}
    }

    #[test]
    fn test_edgelist_add_and_query_prefix_support() {
	let trie = Trie::new_with_edgelist();
	add_and_query_prefix_support( trie );
    }

        #[test]
    fn test_better_edgelist_add_and_query_prefix_support() {
	let trie = Trie::new_with_edgelist_better();
	add_and_query_prefix_support( trie );
    }

    #[test]
    fn test_skipgraph_add_and_query_prefix_support() {
	let trie = Trie::new_with_skipgraph( 3 );
	add_and_query_prefix_support( trie );
    }
}
