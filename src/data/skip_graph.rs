
use std::collections::HashMap;

use bit_set::BitSet;

/// Stores a list of sequences as a graph.  A sequence is represented by its index in the list.
/// There is a node for every symbol in the sequences. A node is represented by its index.  The set of nodes is fixed on construction.
/// There is an edge (a,b,s) if a < b and s is a sequence where b occurs after a
struct SkipGraph {
    /// Fixed number of nodes
    number_nodes: usize,
    /// multi-edges between symbols
    edges: Vec<EdgeMap>,
    /// Store which sequences have which initial symbol
    start_symbols: Vec<BitSet>,
    /// Terminal node per sequence
    sequence_to_stop: Vec<usize>,
    /// Number of sequences stored
    number_sequences: usize,
}

type EdgeMap = HashMap<usize, BitSet>;

impl SkipGraph {

    pub fn new( num_nodes: usize ) -> SkipGraph {
	SkipGraph{ 
	    number_nodes: num_nodes,
	    edges: vec![EdgeMap::new(); num_nodes],
	    start_symbols: vec![BitSet::new(); num_nodes],
	    sequence_to_stop: vec!(),
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
	assert!( initial_node < self.number_nodes );
	self.start_symbols[ initial_node ].insert( seqid );
    }
    
    /// Returns the set of sequences that have the prefix.
    pub fn get_sequence_with_prefix( &self, prefix: &[usize] ) -> BitSet {
	let mut prefix_set = self.get_start_set( prefix );
	// for pos in 1 .. prefix.len() {

	// }
	prefix_set
    }

    // /// Returns the set of sequences that have the subsequence
    // pub fn get_sequence_with_subsequence( &self, subsequence: &[T] ) -> BitSet {

    // }

    fn get_start_set( &self, prefix: &[usize] ) -> BitSet {
	if prefix.is_empty() {
	    let mut all_in = BitSet::with_capacity( self.number_nodes );
	    for i in 0 .. self.number_nodes {
		all_in.insert( i );
	    }
	    return all_in;
	}
	let first_node = prefix[0];
	self.start_symbols[ first_node ].clone()
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

	let prefix_set = skipgraph.get_sequence_with_prefix( &[] );
	assert!( prefix_set.contains( 0 ));
	assert!( prefix_set.contains( 1 ));
    }


}
