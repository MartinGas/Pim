
use std::collections::HashMap;
use std::cmp::max;

use bit_set::BitSet;
use bit_vec::BitVec;

use super::ItemVec;

/// Stores a list of sequences as a graph.  A sequence is represented by its index in the list.
/// There is a node for every symbol in the sequences. A node is represented by its index.  The set of nodes is fixed on construction.
/// There is an edge (a,b,s) if a < b and s is a sequence where b occurs after a
pub struct SkipGraph {
    /// nodes labeled with sequences that contain them
    nodes: Vec<BitSet>,
    /// Store which sequences have which initial symbol
    first_nodes: Vec<BitSet>,
    /// Store the last symbol of for every sequence
    /// Invariant: index coincides with sequence id
    last_nodes: Vec<Node>,
    /// multi-edges between symbols
    edges: Vec<EdgeMap>,
    /// Stores edge to the directly succeeding node for every sequence
    first_edges: Vec<EdgeMap>,
    /// Number of sequences stored
    number_sequences: usize,
}

pub struct SkipGraphIterator<'a> {
    first_nodes: &'a Vec<BitSet>,
    first_edges: &'a Vec<EdgeMap>,
    start_node: usize,
    sequence_iterator: <&'a BitSet as IntoIterator>::IntoIter,
}

type Node = usize;
type EdgeMap = HashMap<Node, BitSet>;
type Sequence = usize;

impl SkipGraph {

    pub fn new( num_nodes: usize ) -> SkipGraph {
	SkipGraph{ 
	    nodes: vec![BitSet::new(); num_nodes],
	    first_nodes: vec![BitSet::new(); num_nodes],
	    last_nodes: Vec::new(),
	    edges: vec![EdgeMap::new(); num_nodes],
	    first_edges: vec![ EdgeMap::new(); num_nodes ],
	    number_sequences: 0,
	}
    }

    /// Adds the sequence to the graph
    /// Pre: sequence is not empty
    /// Pre: sequence only contains nodes in the graph
    pub fn add( &mut self, sequence: &[usize] ) -> Sequence {
	assert!( !sequence.is_empty() );
 
	let seqid = self.number_sequences;
	self.number_sequences += 1;

	let initial_node = sequence[0];
	self.first_nodes[ initial_node ].insert( seqid );
	let last_node = sequence.last().expect( "non-empty sequence" );
	self.last_nodes.push( *last_node );
	
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
	seqid
    }

    /// Cuts a sequence off at a specified index s.t. element at index is the last element before the cut
    /// Thus, this method cannot remove a sequence completely.
    /// Returns the truncated suffix.
    /// Pre: sequence_identifier is a valid index
    pub fn truncate( &mut self, sequence_identifier: usize, cut_index: usize ) -> Vec<Node> {
	assert!( sequence_identifier < self.number_sequences );

	// reconstruct the sequence
	let sequence = self.reconstruct( sequence_identifier );
	// last node is earlier now
	self.last_nodes[ sequence_identifier ] = sequence[ cut_index ];
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

	sequence[ cut_index + 1 ..].iter().copied().collect()
    }

    /// Reconstructs the sequence by following its first edges.
    /// Note: This operation is expensive, because we need to search the next symbols for the sequence's first edge and first node.
    /// Pre: sequence_identifier refers to a valid sequence
    pub fn reconstruct( &self, sequence_identifier: usize ) -> Vec<Node> {
	let mut sequence = Vec::new();
	let mut node = self.first_nodes.iter().enumerate()
	    .find( |(_node, seqset)| seqset.contains( sequence_identifier ))
	    .unwrap().0; // sequence_identifier is valid and every sequence is non-empty

	// follow the first edges labeled with seqid
	loop {
	    sequence.push( node ); 
	    let next_node = self.first_edges[ node ].iter()
		.find( |(_to_node, seqset)| seqset.contains( sequence_identifier ));
	    if next_node.is_none() { // no further edge
		return sequence;
	    }

	    node = *next_node.unwrap().0;
	}
    }
    
    /// Returns a map of sequences with part of the prefix as their prefix.
    /// Sequence IDs map to the number of matching nodes and the indicator for the completeness of the prefix.
    pub fn get_sequence_with_prefix( &self, prefix: &[usize] ) -> HashMap<Sequence, (usize, bool)> {
	let mut sequence_to_num_nodes: HashMap<Sequence, (usize, bool)> = HashMap::new();
	// get indicators for start node (no jump)
	let mut prefix_set = match prefix.get( 0 ) {
	    None => self.build_all_in_set(),
	    Some( start ) => self.first_nodes[ *start ].clone(),
	};
	// traverse the graph along the first edges matching prefix
	for pos in 1 .. prefix.len() {
	    if prefix_set.is_empty() {
		return sequence_to_num_nodes;
	    }
	    
	    let from_node = prefix[ pos - 1 ];
	    let to_node = prefix[ pos ];

	    // Determine sequences that fail to match. If there's no edge, all fail.
	    let unmatching = match self.first_edges[ from_node ].get( &to_node ) {
		Some( edge_label ) => {
		    let mut unmatching = prefix_set.clone();
		    unmatching.difference_with( edge_label );
		    unmatching
		},
		None => prefix_set.clone(),
	    };

	    // Remove all unmatching and save as result.
	    prefix_set.difference_with( &unmatching );
	    for seqid in unmatching.iter() {
		// If the sequences stopped before the prefix ended, the prefix is complete.
		let is_done = self.last_nodes[ seqid ] == from_node;
		sequence_to_num_nodes.insert( seqid, (pos, is_done) );
	    }
	}

	// Add all complete prefixes to the map.
	for seqid in prefix_set.iter() {
	    sequence_to_num_nodes.insert( seqid, (prefix.len(), true) );
	}

	sequence_to_num_nodes
    }

    /// Returns a map that gives the number of nodes visited for every sequence that has a maximal prefix of the subsequence.
    /// A sequence has a maximal prefix if it has all elements of the subsequence that are smaller or equal to its largest element.
    pub fn get_sequence_with_subsequence( &self, subsequence: &[Node] ) -> HashMap<Sequence, usize> {
	// maintain which sequences are subsequences and how many nodes were visited
	let mut sequence_to_num_nodes: HashMap<Sequence, usize> = HashMap::new();
	// maintain which sequences remain to be checked
	let mut open_set = match subsequence.get( 0 ) {
	    None => self.build_all_in_set(),
	    Some( front ) => {
		// All sequences that are closed and whose last item is smaller than front have the subsequence too.
		for (seqid, last_node) in self.last_nodes.iter().enumerate() {
		    if last_node < front {
			sequence_to_num_nodes.insert( seqid, 0 );
		    }
		}
		self.nodes[ *front ].clone()
	    }
	};



	// traverse the prefix until no more paths are available
	for pos in 1 .. subsequence.len() {
	    if open_set.is_empty() { // no more sequences to check
		return sequence_to_num_nodes;
	    }

	    let from_node = subsequence[ pos - 1 ];
	    let to_node = subsequence[ pos ];

	    // close all sequences whose last element is smaller than to_node
	    let mut sequences_closed = BitSet::with_capacity( open_set.capacity() );
	    for seqid in open_set.iter() {
		if self.last_nodes[ seqid ] < to_node {
		    sequences_closed.insert( seqid );
		}
	    }
	    for seqid in sequences_closed.iter() {
		open_set.remove( seqid );
		// visited pos nodes before closing
		sequence_to_num_nodes.insert( seqid, pos );
	    }

	    let edge_labels = self.edges[ from_node ].get( &to_node );
	    if edge_labels.is_none() {
		open_set.clear(); // no path forward
	    } else {
		// remove sequences that do not match
		open_set.intersect_with( &edge_labels.unwrap() );
	    }
	}

	// Add all sequences that are still open. We visited the entire subsequence.
	// Handles the case where subsequence is empty.
	for seqid in open_set.iter() {
	    sequence_to_num_nodes.insert( seqid, subsequence.len() );
	}
	
	sequence_to_num_nodes
    }

    fn build_all_in_set( &self ) -> BitSet {
	let num_nodes = self.nodes.len();
	let vec = BitVec::from_elem( num_nodes, true );
	BitSet::from_bit_vec( vec )
    }
}

impl <'a> IntoIterator for &'a SkipGraph {
    type Item = <SkipGraphIterator<'a> as Iterator>::Item;
    type IntoIter = SkipGraphIterator<'a>;

    fn into_iter( self ) -> Self::IntoIter {
	SkipGraphIterator::new( &self.first_nodes, &self.first_edges )
    }
}

impl <'a> SkipGraphIterator<'a> {

    pub fn new( first_nodes: &'a Vec<BitSet>, first_edges: &'a Vec<EdgeMap> ) -> SkipGraphIterator<'a> {
	SkipGraphIterator{ first_nodes, first_edges,
			   start_node: 0,
			   sequence_iterator: first_nodes[ 0 ].iter() }
    }

    /// Attempts to advance the top most iterator. Pops the stack until some iterator is not exhausted.
    fn advance( &mut self ) -> Option<Sequence> {
	// look for the next sequence
	let next_seqid = self.sequence_iterator.next();
	if let Some( seqid ) = next_seqid {
	    return Some( seqid );
	}

	// Processed all sequences for the current node. Go to the next one.
	self.start_node += 1;
	if !(self.start_node < self.first_nodes.len()) { // Processed all nodes.
	    return None;
	}

	// reset sequence iterator and recurse
	self.sequence_iterator = self.first_nodes[ self.start_node ].iter();
	self.advance()
    }

    fn follow( &self, seqid: Sequence ) -> ItemVec {
	let mut sequence = Vec::new();
	let mut node = self.start_node;
	// jump along the first edge for the given sequence
	loop {
	    sequence.push( node );
	    let next_node = self.first_edges[ node ].iter()
		.find_map( |(node, seqset)| if seqset.contains( seqid ) { Some( node ) } else { None } );

	    match next_node {
		Some( next_node ) => node = *next_node,
		None => return sequence,
	    }
	}
    }
}

impl <'a> Iterator for SkipGraphIterator<'a> {
    type Item = (Sequence, ItemVec);

    fn next( &mut self ) -> Option<Self::Item> {
	let next_seqid = self.advance();
	match next_seqid {
	    Some( seqid ) => Some( (seqid, self.follow( seqid )) ),
	    None => None,
	}
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashSet;
    
    #[test]
    fn test_empty_prefix() {
	let num_nodes = 3;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1 ));
	skipgraph.add( &vec!( 0, 2 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( *prefix_set.get( &0 ).expect( "contains key 0" ), (0, true) );
	assert!( prefix_set.contains_key( &1 ));
	assert_eq!( *prefix_set.get( &1 ).expect( "contains key 1" ), (0, true) );
    }

    #[test]
    fn test_nonempty_prefix() {
	let num_nodes = 4;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1 ));
	skipgraph.add( &vec!( 0, 2 ));
	skipgraph.add( &vec!( 2, 3 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 1] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( *prefix_set.get( &0 ).expect( "contains key 0" ), (2, true) );
	assert!( prefix_set.contains_key( &1 ));
	assert_eq!( *prefix_set.get( &1 ).expect( "contains key 1" ), (1, false) );
	assert!( !prefix_set.contains_key( &2 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[1, 2] );
	assert!( prefix_set.is_empty() );

	let prefix_set = skipgraph.get_sequence_with_prefix( &[2, 3] );
	assert!( prefix_set.contains_key( &2 ));
	assert_eq!( *prefix_set.get( &2 ).expect( "contains key 2" ), (2, true) );
	assert!( !prefix_set.contains_key( &1 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0] );
	assert_eq!( prefix_set.len(), 2 );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( prefix_set[ &0 ], (1, true) );
	assert!( prefix_set.contains_key( &1 ));
	assert_eq!( prefix_set[ &1 ], (1, true) );
    }

    #[test]
    fn test_larger_prefix() {
	let mut skipgraph = SkipGraph::new( 4 );
	skipgraph.add( &vec!( 0 ));
	skipgraph.add( &vec!( 0, 1 ));

	// test with a larger prefix than any of the sequences
	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 5] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( *prefix_set.get( &0 ).expect( "contains key 0" ), (1, true) );
	assert!( prefix_set.contains_key( &1 ));
	assert_eq!( *prefix_set.get( &1 ).expect( "contains key 1" ), (1, false) );

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 1, 5] );
	assert!( prefix_set.contains_key( &1 ));
	assert_eq!( *prefix_set.get( &1 ).expect( "contains key 0" ), (2, true) );
    }

    #[test]
    /// Test whether the terminal node is used correctly to determine if a sequence is closed
    fn test_touching_prefix() {
	let mut skipgraph = SkipGraph::new( 4 );
	skipgraph.add( &vec!( 0, 1, 2 ));
	// skipgraph.add( &vec!( 0, 2, 3 ));

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 2, 3] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( prefix_set[ &0 ], (1, false) );

	let prefix_set = skipgraph.get_sequence_with_prefix( &[0, 3] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( prefix_set[&0], (1, false) );
    }

    #[test]
    /// If we truncate a larger sequence, some traces are left.
    /// Check that we still correctly determine whether a sequence has ended
    fn test_truncate_prefix() {
	let mut graph = SkipGraph::new( 3 );
	graph.add( &vec!( 0, 1, 2 ));
	graph.truncate( 0, 0 );

	// After truncating past index 0 we are left with sequence [0] so [0,1] is a complete partial prefix.
	let prefix_set = graph.get_sequence_with_prefix( &[0, 1] );
	assert!( prefix_set.contains_key( &0 ));
	assert_eq!( prefix_set[ &0 ], (1, true) );
    }

    #[test]
    fn test_empty_subsequence() {
	let num_nodes = 3;
	let mut skipgraph = SkipGraph::new( num_nodes );
	skipgraph.add( &vec!( 0, 1, 2 ));

	let result = skipgraph.get_sequence_with_subsequence( &[] );
	assert!( result.contains_key( &0 ) );
	assert_eq!( *result.get( &1 ).unwrap(), 0 );
    }

    #[test]
    fn test_nonempty_subsequence() {
	let num_nodes = 4;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1, 2 ));
	graph.add( &vec!( 1, 2, 3 ));

	let subs = graph.get_sequence_with_subsequence( &[1, 2] );
	println!( "{subs:?}" );
	assert!( subs.contains_key( &0 ));
	assert_eq!( *subs.get( &0 ).unwrap(), 2 );
	assert!( subs.contains_key( &1 ));
	assert_eq!( *subs.get( &1 ).unwrap(), 2 );

	let subs = graph.get_sequence_with_subsequence( &[0, 3] );
	assert!( subs.contains_key( &0 )); // 0 is closed before 3 occurs
	assert_eq!( *subs.get( &0 ).unwrap(), 1 );
    }

    #[test]
    fn test_non_overlapping_greater_subsequence() {
	let num_nodes = 5;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1, 2 ));

	let subs = graph.get_sequence_with_subsequence( &[3, 4] );
	assert!( subs.contains_key( &0 ));
	assert_eq!( subs[ &0 ], 0 );
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
	assert!( subseq.contains_key( &0 ));

	// part after truncation is gone
	let subseq = graph.get_sequence_with_subsequence( &[1, 0] );
	assert!( subseq.is_empty() );
    }

    #[test]
    fn test_iterate() {
	let num_nodes = 3;
	let mut graph = SkipGraph::new( num_nodes );
	graph.add( &vec!( 0, 1 ));
	graph.add( &vec!( 1, 2 ));

	let sequences: HashSet<(Sequence, ItemVec)> = graph.into_iter().collect();
	assert_eq!( sequences.len(), 2 );
	assert!( sequences.contains( &(0, vec!( 0, 1 )) ));
	assert!( sequences.contains( &(1, vec!( 1, 2 )) ));
    }


}
