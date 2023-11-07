
use std::collections::HashMap;
use std::cmp::max;

use bit_set::BitSet;
use bit_vec::BitVec;

/// Stores a list of sequences as a graph.  A sequence is represented by its index in the list.
/// There is a node for every symbol in the sequences. A node is represented by its index.  The set of nodes is fixed on construction.
/// There is an edge (a,b,s) if a < b and s is a sequence where b occurs after a
struct SkipGraph {
    /// nodes labeled with sequences that contain them
    nodes: Vec<BitSet>,
    /// Store which sequences have which initial symbol
    first_symbols: Vec<BitSet>,
    /// multi-edges between symbols
    edges: Vec<EdgeMap>,
    /// Stores edge to the directly succeeding node for every sequence
    first_edges: Vec<EdgeMap>,
    /// Number of sequences stored
    number_sequences: usize,
}

type Node = usize;
type EdgeMap = HashMap<Node, BitSet>;

impl SkipGraph {

    pub fn new( num_nodes: usize ) -> SkipGraph {
	SkipGraph{ 
	    nodes: vec![BitSet::new(); num_nodes],
	    first_symbols: vec![BitSet::new(); num_nodes],
	    edges: vec![EdgeMap::new(); num_nodes],
	    first_edges: vec![ EdgeMap::new(); num_nodes ],
	    number_sequences: 0,
	}
    }

    /// Adds the sequence to the graph
    /// Pre: sequence is not empty
    /// Pre: sequence only contains nodes in the graph
    pub fn add( &mut self, sequence: &[usize] ) {
	assert!( !sequence.is_empty() );

	let seqid = self.number_sequences;
	self.number_sequences += 1;

	let initial_node = sequence[0];
	self.first_symbols[ initial_node ].insert( seqid );
	
	// add the edges for all successors
	for pos in 0 .. sequence.len() - 1 {
	    let node = sequence[ pos ];
	    self.nodes[ node ].insert( seqid );
	    
	    for successor in &sequence[pos + 1 ..] {
		let edge_labels = self.edges[ node ].entry( *successor ).or_default();
		edge_labels.insert( seqid );
	    }

	    // add first edge
	    let next_node = sequence[ pos + 1 ];
	    let first_edge_labels = self.first_edges[ node ].entry( next_node ).or_default();
	    first_edge_labels.insert( seqid );
	}
    }

    /// Cuts a sequence off at a specified index s.t. element at index is the last element before the cut
    /// Thus, this method cannot remove a sequence completely.
    /// Pre: sequence_identifier is a valid index
    pub fn truncate( &mut self, sequence_identifier: usize, cut_index: usize ) {
	assert!( sequence_identifier < self.number_sequences );

	// reconstruct the sequence
	let sequence = self.reconstruct( sequence_identifier );
	// remove all edges for this sequence past the cut
	for (pos, from_node) in sequence.iter().enumerate() {
	    let start = max( pos + 1, cut_index + 1 );
	    for to_node in sequence[ start .. ].iter() {
		self.edges[ *from_node ].get_mut( to_node ).expect( "reconstruction from edges" ).remove( sequence_identifier );
	    }
	}
	// remove all first edges past the cut
	for (pos, from_node) in sequence[ cut_index .. sequence.len() - 1 ].iter().enumerate() {
	    let to_node = sequence[ cut_index + pos + 1 ];
	    self.first_edges[ *from_node ].get_mut( &to_node ).expect( "reconstruction from edges" ).remove( sequence_identifier );
	}
    }

    /// Reconstructs the sequence by following its first edges.
    /// Note: This operation is expensive, because we need to search the next symbols for the sequence's first edge and first node.
    /// Pre: sequence_identifier refers to a valid sequence
    pub fn reconstruct( &self, sequence_identifier: usize ) -> Vec<Node> {
	let mut sequence = Vec::new();
	let mut node = self.first_symbols.iter().enumerate()
	    .find( |(node, seqset)| seqset.contains( sequence_identifier ))
	    .unwrap().0; // sequence_identifier is valid and every sequence is non-empty

	// follow the first edges labeled with seqid
	loop {
	    sequence.push( node ); 
	    let next_node = self.first_edges[ node ].iter()
		.find( |(to_node, seqset)| seqset.contains( sequence_identifier ));
	    if next_node.is_none() { // no further edge
		return sequence;
	    }

	    node = *next_node.unwrap().0;
	}
    }
    
    /// Returns the set of sequences that have the prefix and the number of nodes visited.
    pub fn get_sequence_with_prefix( &self, prefix: &[usize] ) -> (BitSet, usize) {
	// get indicators for start node (no jump)
	let prefix_set = match prefix.get( 0 ) {
	    None => self.build_all_in_set(),
	    Some( start ) => self.first_symbols[ *start ].clone(),
	};
	// traverse the graph along the first edges matching prefix
	self.traverse_by( prefix_set, prefix, |graph, left, right| graph.first_edges[ left ].get( &right ))
    }

    /// Returns the set of sequences that have the subsequence
    pub fn get_sequence_with_subsequence( &self, subsequence: &[Node] ) -> BitSet {
	// get indicators for first initial element of subsequence
	let subseq_set = match subsequence.get( 0 ) {
	    None => self.build_all_in_set(),
	    Some( front ) => self.nodes[ *front ].clone(),
	};
	// traverse the graph along any edge matching prefix
	self.traverse_by( subseq_set, subsequence, |graph, left, right| graph.edges[ left ].get( &right ) ).0
    }

    /// Traverses the graph along the edges provided by the edge function for the given initial selection and remaining sequence
    /// Returns the set of sequences that match the traversal and the number of nodes visited.
    fn traverse_by <'a, F> ( &'a self, initial_selection: BitSet, sequence: &[Node], edge_function: F ) -> (BitSet, usize) where F: Fn(&'a Self, Node, Node) -> Option<&'a BitSet> {
	let mut selection = initial_selection;
	let mut visits = if sequence.is_empty() { 0 } else { 1 };
	// iterate over adjacent pairs of the sequence
	for pos in 1 .. sequence.len() { 
	    if selection.is_empty() { // no contiguous paths, return early
		return (selection, visits);
	    }

	    let from_node = sequence[ pos - 1 ];
	    let to_node = sequence[ pos ];
	    let edge_labels = edge_function( &self, from_node, to_node );
	    if edge_labels.is_none() {
		return (BitSet::new(), visits); // no path forward
	    }
	    
	    selection.intersect_with( edge_labels.unwrap() );
	    visits += 1;
	}
	(selection, visits)
    }

    fn build_all_in_set( &self ) -> BitSet {
	let num_nodes = self.nodes.len();
	let vec = BitVec::from_elem( num_nodes, true );
	BitSet::from_bit_vec( vec )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_empty_prefix() {
	let num_nodes = 3;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1 ));
	skipgraph.add( &vec!( 0, 2 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[] ).0;
	assert!( prefix_set.contains( 0 ));
	assert!( prefix_set.contains( 1 ));
    }

    #[test]
    fn test_nonempty_prefix() {
	let num_nodes = 4;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1 ));
	skipgraph.add( &vec!( 0, 2 ));
	skipgraph.add( &vec!( 2, 3 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 1] );
	assert!( prefix_set.0.contains( 0 ));
	assert!( !prefix_set.0.contains( 1 ));
	assert!( !prefix_set.0.contains( 2 ));
	assert_eq!( prefix_set.1, 2 );

	let prefix_set = skipgraph.get_sequence_with_prefix( &[1, 2] );
	assert!( prefix_set.0.is_empty() );
	assert_eq!( prefix_set.1, 1 );

	let prefix_set = skipgraph.get_sequence_with_prefix( &[2, 3] );
	assert!( prefix_set.0.contains( 2 ));
	assert!( !prefix_set.0.contains( 1 ));
	assert_eq!( prefix_set.1, 2 );
    }

    #[test]
    fn test_empty_subsequence() {
	let num_nodes = 3;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1, 2 ));

	assert!( skipgraph.get_sequence_with_subsequence( &[] ).contains( 1 ));
    }

    #[test]
    fn test_nonempty_subsequence() {
	let num_nodes = 4;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1, 2 ));
	graph.add( &vec!( 1, 2, 3 ));

	let subs = graph.get_sequence_with_subsequence( &[1, 2] );
	assert!( subs.contains( 0 ));
	assert!( subs.contains( 1 ));

	let subs = graph.get_sequence_with_subsequence( &[0, 3] );
	assert!( subs.is_empty() );
    }

    #[test]
    fn test_reconstruct() {
	let num_nodes = 3;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1, 2 ));
	graph.add( &vec!( 1, 2 ));

	assert_eq!( graph.reconstruct( 0 ), vec!( 0, 1, 2 ));
	assert_eq!( graph.reconstruct( 1 ), vec!( 1, 2 ));
    }
    
    #[test]
    fn test_truncate() {
	let num_nodes = 4;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1, 2, 3 ));
	graph.truncate( 0, 1 );

	// part before truncation is intact
	let subseq = graph.get_sequence_with_prefix( &[0, 1] );
	assert!( subseq.0.contains( 0 ));

	// part after truncation is gone
	let subseq = graph.get_sequence_with_subsequence( &[1, 0] );
	assert!( subseq.is_empty() );
    }


}
