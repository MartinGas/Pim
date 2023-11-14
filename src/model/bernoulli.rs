
mod serialize; // serialization and pretty printing of the model

use std::collections::{HashSet, HashMap};
use bit_set::BitSet;
use statrs::distribution::{Beta, Continuous};
use rayon::prelude::*;

use crate::DataPair;

use super::*;

pub use serialize::BernoulliFormatter;

#[derive( Debug )]
/// Bernoulli model with pattern assignments as latent variables and additive and destructive noise
pub struct BernoulliAssignment {
    // add patterns later
    // add_item_noise: HashMap<Item, f64>,
    // kill_item_noise: HashMap<Item, f64>,
    // pattern_prob: HashMap<Token, f64>,
    universe: Vec<Item>,
    patterns: HashMap<Token, InternalPattern>,
    parameters: Parameters,
    token_count: usize,
}

// \todo move these structs to sub-modules

#[derive( Debug, Clone )]
/// Manages the parameters of the Bernoulli model.
pub struct Parameters {
    /// Bernoulli parameters for non-covered items
    add_item_noise: HashMap<Item, f64>,
    /// prior distribution over additive noise parameter
    add_item_prior: Beta,
    /// Bernoulli parameters for covered items
    kill_item_noise: HashMap<Item, f64>,
    /// prior distribution over destructive noise parameter
    kill_item_prior: Beta,
    /// Bernoulli parameters for patterns
    pattern_prob: HashMap<Token, f64>,
    /// prior distribution over pattern parameter
    pattern_prior: Beta,
    /// number of transactions observed
    number_transactions: Count,
}

#[derive( Debug )]
pub struct UseCount {
    /// Every item stores the counts [occur_cover, occur_not_cover, not_occur_cover, not_occur_not_cover]
    item_cover_table: HashMap<Item, [Count; 4]>,
    pattern_count: HashMap<Token, Count>,
    universe: Vec<Item>,
    number_transactions: Count,
}

#[derive( Debug )]
/// Stores pattern and estimated probability of this pattern in the augmented model.
pub struct CandidatePattern {
    pattern: InternalPattern,
    gain_estimate: f64,
    probability_estimate: f64,
    realization: Option<Token>,
}

/// Generates patterns given a view of the model's parameters
pub struct PatternRecombinator {
    // Stores candidates
    candidate_queue: Vec<CandidatePattern>,
    // Stores patterns that may not be candidates
    blocked: HashSet<InternalPattern>,
    // bias and current parameters
    parameters: Parameters,
}

// representation of the pattern's contents
type InternalPattern = Vec<usize>;
type PatternPair<'a> = (Token, &'a InternalPattern);

impl Model for BernoulliAssignment {
    type Cover = UseCount;
    type Candidate = CandidatePattern;
    
    fn cover <'a, I> ( &self, data: I ) -> Self::Cover where
	I: Iterator<Item = DataPair>,
    {
	let mut cover = UseCount::new( self.universe.clone() );
	for (transaction, count) in data {
	    let patterns = self.cover_transaction_greedy( &transaction );
	    cover.add_cover( &transaction, &patterns, count );
	}
	cover
    }

    fn fit( &mut self, cover: &Self::Cover ) {
	debug!( "Parameters before fit {:?}", self.parameters.pattern_prob );

	// old parameters are invalid
	self.parameters.clear();
	
	let n = cover.number_transactions;
	self.parameters.set_number_transactions( n );
	for item in &self.universe {
	    let occur_not_cover_count = cover.get_item_cover_count( *item, true, false );
	    let not_occur_not_cover_count = cover.get_item_cover_count( *item, false, false );
	    self.parameters.fit_additive_noise_map( *item, occur_not_cover_count, not_occur_not_cover_count );

	    let not_occur_cover_count = cover.get_item_cover_count( *item, false, true );
	    let occur_cover_count = cover.get_item_cover_count( *item, true, true );
	    self.parameters.fit_destructive_noise_map( *item, not_occur_cover_count, occur_cover_count );
	}
	for token in self.patterns.keys() {
	    let usage = cover.get_pattern_count( *token );
	    self.parameters.fit_pattern_prob_map( *token, usage, n - usage );
	}

	debug!( "Parameter after fit {:?}", self.parameters.pattern_prob );
    }

    fn calc_loglik( &self, cover: &UseCount ) -> f64 {
	let mut loglik = 0.0;

	for item in &self.universe {
	    loglik += self.calc_additive_noise_loglik( *item, cover );
	    loglik += self.calc_destructive_noise_loglik( *item, cover );
	    loglik += self.calc_pattern_loglik( *item, cover );
	}
	loglik
    }

    fn generate_candidates <D: Database + Sync> ( &self, data: &D ) -> Box<dyn Iterator<Item = Self::Candidate>> {
	let blocked: HashSet<InternalPattern> = self.patterns.values().cloned().collect();
	let mut recombinator = PatternRecombinator::new( blocked, &self.parameters );
	// todo: can we avoid the copy?
	let mut augmented_patterns: Vec<InternalPattern> = self.patterns.values().cloned().collect();
	for item in &self.universe { // add singletons as patterns
	    augmented_patterns.push( vec!( *item ));
	}
	recombinator.combine_patterns( &augmented_patterns, data );
	recombinator.sort_candidates();
	Box::new( recombinator )
    }

    fn add_candidate( &mut self, candidate: &mut Self::Candidate ) {
	debug!( "Adding candidate {:?} (gain {:?})", candidate.pattern, candidate.gain_estimate );
	let token = self.create_pattern( candidate.pattern.iter().copied() );
	self.parameters.set_pattern_prob( token, candidate.probability_estimate );
	candidate.realization = Some( token );
    }

    fn remove_candidate( &mut self, candidate: &mut Self::Candidate ) {
	if candidate.realization.is_none() {
	    panic!( "The candidate was not added before removal" );
	}
	let token = candidate.realization.take().unwrap();
	self.patterns.remove( &token );
	// will be removed from the parameter set on the next fit
    }
}

impl BernoulliAssignment {

    pub fn new <'a, U> ( universe: U ) -> BernoulliAssignment where U: IntoIterator<Item = &'a Item> {
	// set default bias
	let mut initial = Parameters::new();
	// favor not using noise
	initial.set_additive_bias( 1, 2 );
	initial.set_destructive_bias( 1, 2 );
	// patterns are unbiased so we can get rid of them
	// initial.set_pattern_bias( 1, 1 );

	BernoulliAssignment{
	    // add_item_noise: HashMap::new(),
	    // kill_item_noise: HashMap::new(),
	    // pattern_prob: HashMap::new(),
	    universe: universe.into_iter().map( |itm| *itm ).collect(),
	    patterns: HashMap::new(),
	    parameters: initial,
	    token_count: 0,
	}
    }

    pub fn create_pattern <'a, I> ( &mut self, items: I ) -> Token where I: Iterator<Item = Item> {
	let pattern: InternalPattern = items.collect();
	let token = self.token_count;
	self.token_count += 1;
	self.patterns.insert( token, pattern );
	token
    }

    pub fn iterate_universe<'a>( &'a self ) -> Box<dyn Iterator<Item = Item> + 'a> {
	Box::new( self.universe.iter().copied() )
    }
    
    pub fn iterate_tokens<'a>( &'a self ) -> Box<dyn Iterator<Item = Token> + 'a> {
	Box::new( self.patterns.keys().copied() )
    }
    
    pub fn get_pattern( &self, pattern: Token ) -> Option<&InternalPattern> {
	self.patterns.get( &pattern )
    }

    pub fn get_parameters( &self ) -> &Parameters {
	&self.parameters
    }

    // \deprecated
    // todo: remove
    pub fn calc_additive_noise_loglik( &self, item: Item, cover: &UseCount ) -> f64 {
	let not_occur_not_cover = cover.get_item_cover_count( item, false, false );
	let occur_not_cover = cover.get_item_cover_count( item, true, false );
	self.parameters.calc_additive_loglik( item, occur_not_cover, not_occur_not_cover )
    }

    // \deprecated
    // todo: remove
    pub fn calc_destructive_noise_loglik( &self, item: Item, cover: &UseCount ) -> f64 {
	let not_occur_cover = cover.get_item_cover_count( item, false, true );
	let occur_cover = cover.get_item_cover_count( item, true, true );
	self.parameters.calc_destructive_loglik( item, not_occur_cover, occur_cover )
    }

    // \deprecated
    // todo: remove
    pub fn calc_pattern_loglik( &self, pattern: Token, cover: &UseCount ) -> f64 {
	let usage = cover.get_pattern_count( pattern );
	let n = cover.get_transaction_count();
	self.parameters.calc_pattern_loglik( pattern, usage, n - usage )
    }

    pub fn cover_transaction_greedy <'a, 'b> ( &'a self, t: &'b Transaction ) -> Vec<PatternPair> {
	// todo: re-computing the best pattern has quadratic complexity in the number of patterns
	// We could use a treap instead to maange patterns/tokens and priorities/cost.
	// Assuming that a change in one item only affects a constant number of patterns, the total cost decreases to n*log(n),
	// because we can locate and update pattern costs in time log(n).
	// todo: move to its own struct, so we can take disentagle this function

	// Observe that pattern and destructive noise costs can be calculated upfront.
	// Only the savings on additive noise items change.
	let mut gain_per_pattern: HashMap<Token, f64> = HashMap::new();
	let mut cover_candidates: HashSet<Token> = HashSet::new();
	let mut item_to_pattern: HashMap<Item, HashSet<Token>> = HashMap::new();
	for token in self.patterns.keys() {
	    let pattern = self.get_pattern( *token ).expect( "patterns are internally consistent" );
	    let gain = self.calc_empty_cover_gain( t, *token );
	    if !( gain > 0.0 ) {
		continue;
	    }

	    gain_per_pattern.insert( *token, gain );
	    cover_candidates.insert( *token );
	    for item in pattern {
		item_to_pattern.entry( *item )
		    .and_modify( |group| { group.insert( *token ); } )
		    .or_insert_with( || HashSet::from( [*token] ) );
	    }
	}

	// greedily add candidates to covering
	let mut cover = Transaction::new();
	let mut covering: Vec<PatternPair> = Vec::new();
	while !cover_candidates.is_empty() {
	    // find the best candidate
	    let best_entry = gain_per_pattern.iter().max_by( |left_entry, right_entry| left_entry.1.partial_cmp( right_entry.1 ).expect( "not nan" ) )
		.expect( "there are candidates" );
	    let (best_token, best_gain) = (*best_entry.0, *best_entry.1);
	    // stop if there's no more improvement
	    if !(best_gain > 0.0) {
		break;
	    }

	    let pattern = self.get_pattern( best_token ).expect( "tokens are consistent" );
	    // update costs per added item
	    let uncovered_items = pattern.iter().filter( |itm| !cover.contains( **itm )).copied();
	    for itm in uncovered_items  {
		let offset_saving = self.parameters.calc_additive_loglik( itm, 1, 0 );
		let item_group = item_to_pattern.get_mut( &itm );
		if item_group.is_none() {
		    continue;
		}

		let item_group = item_group.unwrap();
		item_group.remove( &best_token );
		// reduce savings by newly added items
		for token in item_group.iter() {
		    let gain_ref = gain_per_pattern.get_mut( token ).expect( "tokens are consistent" );
		    // offset saving is negative
		    *gain_ref += offset_saving ;
		}
	    }
	    // separate mutation
	    for item in pattern {
		cover.insert( *item );
	    }
	    // keep books on patterns
	    covering.push( (best_token, pattern) );
	    cover_candidates.remove( &best_token );
	    gain_per_pattern.remove( &best_token );
	}
	covering
    }

    /// Calculates the gain of covering a transaction with the pattern corresponding to token
    fn calc_empty_cover_gain( &self, transaction: &Transaction, token: Token ) -> f64 {
	let pattern = self.patterns.get( &token ).expect( "valid token refers to a pattern" );
	// we consider the cost given fixed parameters, so the prior does not matter
	// println!( "Calc empty cover gain parameters {:?}", self.parameters.pattern_prob );

	let pattern_cost = self.parameters.calc_pattern_loglik( token, 1, 0 ) - self.parameters.calc_pattern_loglik( token, 0, 1 );
	let kill_cost: f64 = pattern.iter()
	    .filter( |item| !transaction.contains( **item ))
	    .map( |item| self.parameters.calc_destructive_loglik( *item, 1, 0 )).sum();
	let add_savings: f64 = pattern.iter()
	    .filter( |item| transaction.contains( **item ))
	    .map( |item| self.parameters.calc_additive_loglik( *item, 1, 0 )).sum();

	// question: how do we deal with "infinite" gains? Are they even infinite?
	// Occurs if event with probability 0 happens e.g. item with noise prob 0 occurs
	// should not happen any more with prior over parameters

	pattern_cost + kill_cost - add_savings
    }
}

impl Parameters {

    /// Constructor for model without bias
    pub fn new() -> Parameters {
	Parameters {
	    add_item_noise: HashMap::new(),
	    kill_item_noise: HashMap::new(),
	    pattern_prob: HashMap::new(),
	    // initialize with uniform prior
	    add_item_prior: Beta::new( 1.0, 1.0 ).unwrap(),
	    kill_item_prior: Beta::new( 1.0, 1.0 ).unwrap(),
	    pattern_prior: Beta::new( 1.0, 1.0 ).unwrap(),
	    number_transactions: 0,
	}
    }

    pub fn set_number_transactions( &mut self, number_transactions: Count ) {
	self.number_transactions = number_transactions;
    }
    
    pub fn set_additive_bias( &mut self, positive_bias: Count, negative_bias: Count ) {
	// Shape parameters >= 1.0 can be interpreted as prior "pseudo-samples"
	let a_shape = positive_bias as f64 + 1.0;
	let b_shape = negative_bias as f64 + 1.0;
	self.add_item_prior = Beta::new( a_shape, b_shape ).expect( "provided counts are non-negative" );
    }

    pub fn get_additive_bias( &self ) -> (Count, Count) {
	(self.add_item_prior.shape_a() as Count - 1, self.add_item_prior.shape_b() as Count - 1)
    }
    
    pub fn get_additive_noise( &self, item: Item ) -> f64 {
	let calc_unseen = || {
	    let (bias_on, bias_off) = self.get_additive_bias();
	    calc_prob(bias_on, self.number_transactions + bias_off)
	};
	self.add_item_noise.get( &item ).map_or_else( calc_unseen, |p| *p )
    }

    pub fn set_additive_noise( &mut self, item: Item, probability: f64 ) {
	self.add_item_noise.insert( item, probability );
    }

    /// Fits the Bernoulli parameter for additive noise of the given item
    pub fn fit_additive_noise_mle( &mut self, item: Item, count_on: Count, count_off: Count ) {
	let add_prob = calc_prob( count_on, count_off );
	self.add_item_noise.insert( item, add_prob );
    }

    /// Sets the Bernoulli additive noise parameter to its MAP estimate
    pub fn fit_additive_noise_map( &mut self, item: Item, count_on: Count, count_off: Count ) {
	let (bias_on, bias_off) = self.get_additive_bias();
	let add_prob = calc_prob( count_on + bias_on, count_off + bias_off );
	self.add_item_noise.insert( item, add_prob );
    }

    pub fn calc_additive_loglik( &self, item: Item, on_count: Count, off_count: Count ) -> f64 {
	let prob = self.get_additive_noise( item );
	calc_loglik( prob, on_count, off_count )
    }

    /// Calculates the log of the prior probability of additive noise parameter for the given item
    pub fn calc_additive_logprior( &self, item: Item ) -> f64 {
	let prob = self.get_additive_noise( item );
	self.add_item_prior.ln_pdf( prob ) / f64::ln( 2.0 )
    }

    pub fn set_destructive_bias( &mut self, positive_bias: Count, negative_bias: Count ) {
	let a_shape = positive_bias as f64 + 1.0;
	let b_shape = negative_bias as f64 + 1.0;
	self.kill_item_prior = Beta::new( a_shape, b_shape ).expect( "provide positive shapes" );
    }

    pub fn get_destructive_bias( &self ) -> (Count, Count) {
	(self.kill_item_prior.shape_a() as Count - 1, self.kill_item_prior.shape_b() as Count - 1)
    }

    pub fn get_destructive_noise( &self, item: Item ) -> f64 {
	let calc_unseen = || {
	    let (bias_on, bias_off) = self.get_destructive_bias();
	    // never seen, so never covered too much
	    calc_prob( bias_on, bias_off )
	};
	self.kill_item_noise.get( &item ).map_or_else( calc_unseen, |p| *p )
    }

    pub fn set_destructive_noise( &mut self, item: Item, probability: f64 ) {
	self.kill_item_noise.insert( item, probability );
    }

    pub fn fit_destructive_noise_mle( &mut self, item: Item, count_on: Count, count_off: Count ) {
	let kill_prob = calc_prob( count_on, count_off );
	self.kill_item_noise.insert( item, kill_prob );
    }
    
    pub fn fit_destructive_noise_map( &mut self, item: Item, count_on: Count, count_off: Count ) {
	let (bias_on, bias_off) = self.get_destructive_bias();
	let kill_prob = calc_prob( count_on + bias_on, count_off + bias_off );
	// print!( "Destructive noise {item} {kill_prob:.3}" );
	self.kill_item_noise.insert( item, kill_prob );
    }

    pub fn calc_destructive_loglik( &self, item: Item, on_count: Count, off_count: Count ) -> f64 {
	let prob = self.get_destructive_noise( item );
	calc_loglik( prob, on_count, off_count )
    }

    pub fn calc_destructive_logprior( &self, item: Item ) -> f64 {
	let prob = self.get_destructive_noise( item );
	self.kill_item_prior.ln_pdf( prob ) / f64::ln( 2.0 )
    }

    pub fn set_pattern_bias( &mut self, positive_bias: Count, negative_bias: Count ) {
	let shape_a = positive_bias as f64 + 1.0;
	let shape_b = negative_bias as f64 + 1.0;
	self.pattern_prior = Beta::new( shape_a, shape_b ).expect( "positive shapes" );
    }

    pub fn get_pattern_bias( &self ) -> (Count, Count) {
	(self.pattern_prior.shape_a() as Count - 1, self.pattern_prior.shape_b() as Count - 1)
    }

    pub fn get_pattern_prob( &self, pattern: Token ) -> f64 {
	self.pattern_prob.get( &pattern ).map_or( 0.0, |p| *p )
    }

    pub fn set_pattern_prob( &mut self, pattern: Token, probability: f64 ) {
	self.pattern_prob.insert( pattern, probability );
    }

    /// Removes the parameter for the given pattern
    pub fn forget( &mut self, pattern: Token ) {
	self.pattern_prob.remove( &pattern );
    }

    /// Removes all parameters for patterns
    pub fn clear( &mut self ) {
	self.pattern_prob.clear();
    }

    pub fn fit_pattern_prob_mle( &mut self, pattern: Token, count_on: Count, count_off: Count ) {
	let prob = calc_prob( count_on, count_off );
	self.pattern_prob.insert( pattern, prob );
    }

    pub fn fit_pattern_prob_map( &mut self, pattern: Token, count_on: Count, count_off: Count ) {
	let (bias_on, bias_off) = self.get_pattern_bias();
	let pattern_prob = calc_prob( count_on + bias_on, count_off + bias_off );
	self.pattern_prob.insert( pattern, pattern_prob );
    }

    pub fn calc_pattern_loglik( &self, token: Token, on_count: Count, off_count: Count ) -> f64 {
	let prob = self.get_pattern_prob( token );
	calc_loglik( prob, on_count, off_count )
    }

    pub fn calc_pattern_logprior( &self, token: Token ) -> f64 {
	let prob = self.get_pattern_prob( token );
	self.pattern_prior.ln_pdf( prob ) / f64::ln( 2.0 )
    }
}

fn calc_prob( on_count: Count, off_count: Count ) -> f64 {
    let denominator = on_count + off_count;
    if denominator == 0 { 0.0 }
    else { on_count as f64 / denominator as f64 }
}

/// Calculates the log likelihood of a number of Bernoulli experiments
fn calc_loglik( probability: f64, on_count: Count, off_count: Count ) -> f64 {
    let mut loglik = 0.0;
    loglik += if on_count > 0 { on_count as f64 * log( probability )} else { 0.0 };
    loglik += if off_count > 0 { off_count as f64 * log( 1.0 - probability )} else { 0.0 };
    loglik
}

impl UseCount {
    pub fn new( universe: Vec<Item>) -> UseCount {
	UseCount {
	    item_cover_table: HashMap::new(),
	    pattern_count: HashMap::new(),
	    universe,
	    number_transactions: 0,
	}
    }
    
    pub fn get_item_cover_count( &self, item: Item, is_occur: bool, is_cover: bool ) -> Count {
	// contingency table is_occur x is_cover where true comes before false
	let index = if is_occur { 0 } else { 2 } + if is_cover { 0 } else { 1 };
	self.item_cover_table.get( &item ).map( |table| table[ index ] ).unwrap_or( 0 )
    }
    
    pub fn get_pattern_count( &self, pattern: Token ) -> Count {
	self.pattern_count.get( &pattern ).map_or( 0, |c| *c )
    }

    pub fn get_transaction_count( &self ) -> Count {
	self.number_transactions
    }

    pub fn get_item_support( &self, item: &Item ) -> Count {
	let (cover_index, not_cover_index) = (0, 1);
	self.item_cover_table.get( &item )
	    .map( |table| table[ cover_index ] + table[ not_cover_index ] )
	    .unwrap_or( 0 )
    }

    /// Increments counts for the given covering patterns with the given multiplicity
    pub fn add_cover( &mut self, t: &Transaction, patterns: &[PatternPair], count: Count ) {
	self.number_transactions += count;
	let mut covered = BitSet::with_capacity( t.capacity() );
	for (token, pattern) in patterns {
	    self.pattern_count.entry( *token ).and_modify( |n| *n += count ).or_insert( count );
	    for item in *pattern {
		covered.insert( *item );
	    }
	}

	// increment table counts
	for item in &self.universe {
	    let is_occur = t.contains( *item );
	    let is_cover = covered.contains( *item );
	    let index = if is_occur { 0 } else { 2 } + if is_cover { 0 } else { 1 };
	    let table = self.item_cover_table.entry( *item ).or_insert( [0; 4] );
	    table[ index ] += count;
	}
    }
}

/// Log of base 2 that maps 0.0 to infinity
fn log( x: f64 ) -> f64 {
    if x > 0.0 { f64::log2( x ) } else { f64::NEG_INFINITY }
}

impl CandidatePattern {
    fn new( pattern: InternalPattern, probability_estimate: f64, gain_estimate: f64 ) -> CandidatePattern {
	CandidatePattern{ pattern, probability_estimate, gain_estimate,
			  realization: None,
	}
    }    
}

impl PatternRecombinator {
    fn new( blocked: HashSet<InternalPattern>, current: &Parameters ) -> PatternRecombinator {
	PatternRecombinator{
	    candidate_queue: Vec::new(),
	    blocked,
	    parameters: current.clone(),
	}
    }

    fn combine_patterns<D: Database + Sync> ( &mut self, patterns: &Vec<InternalPattern>, database: &D ) {
	let combinations = patterns.par_iter().enumerate()
	    .flat_map( | (i, left) | {
		patterns[i + 1 ..].par_iter()
		    .map( move | right | (left, right) )
	    });

	let mut new_candidates: Vec<CandidatePattern> = combinations
	    .filter_map( | (left, right) | {
		// todo: combine patterns more efficiently
		let mut pattern = left.clone();
		for item in right {
		    pattern.push( *item );
		}
		pattern.sort();
		pattern.dedup();

		// skip blocked patterns
		if !self.blocked.contains( &pattern ) {
		    let (probability, gain) = self.estimate_gain_over_empty_model( &pattern, database );
		    // println!( "Combine {left:?} + {right:?} = {pattern:?} (gain {gain:.3})" );
		    if gain > 0.0 {
			let candidate = CandidatePattern::new( pattern.clone(), probability, gain );
			// self.candidate_queue.push( candidate );
			// self.blocked.insert( pattern );
			return Some( candidate );
		    }
		}
		None
	    }).collect();
	    
	let range = 0 .. new_candidates.len();
	self.candidate_queue.extend( new_candidates.drain( range ));
    }

    // todo: consider candidates where items are removed
    // This makes only sense, however, if we consider the current covering counts rather than supports.

    /// Constructs the sorted candidate queue. Cleans the candidate table.
    fn sort_candidates( &mut self ) {
	// sort in ascending order, so we can pop at the end
	self.candidate_queue.sort_unstable_by( |left, right| left.gain_estimate.total_cmp( &right.gain_estimate ));
    }

    /// Estimates the pattern probability and gain of adding pattern to the empty model.
    fn estimate_gain_over_empty_model <D: Database> ( &self, pattern: &InternalPattern, database: &D ) -> (f64, f64) {
	let n = database.query_support( vec!() );
	let pattern_support = database.query_support( pattern.clone() );
	
	// todo: consider prior
	let pattern_prob = calc_prob( pattern_support, n - pattern_support );
	let pattern_loglik = calc_loglik( pattern_prob, pattern_support, n - pattern_support );
	let pattern_logprior = 0.0;

	let calc_add_noise_diff = |item: &Item| {
	    let support = database.query_support( vec!( *item ));
	    let new_loglik = self.parameters.calc_additive_loglik( *item, support - pattern_support, n - pattern_support );
	    let old_loglik = self.parameters.calc_additive_loglik( *item, support, n );
	    new_loglik - old_loglik
	};
	let item_diff_sum: f64 = pattern.iter().map( calc_add_noise_diff ).sum();

	trace!( "{pattern:?} : {pattern_prob:.3} est. gain {pattern_loglik:.3} + {pattern_logprior:.3} + {item_diff_sum:.3}" );
	
	(pattern_prob, pattern_loglik + pattern_logprior + item_diff_sum)
    }
}

impl Iterator for PatternRecombinator {
    type Item = CandidatePattern;

    fn next(&mut self) -> Option<Self::Item> {
	self.candidate_queue.pop()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::data::*;

    // is there a better place for this?
    macro_rules! assert_approx {
	($real:expr, $expected:expr, $delta:expr) => {
	    if $real < $expected - $delta || $real > $expected + $delta {
		panic!( "Violate {:.4} == {:.4} (+-{:.4})", $real, $expected, $delta );
	    }
	}
    }

    // macro_rules! bitset { // produces local ambiguity -- what does it mean?
    // 	() => ( BitSet::new(); );
    // 	( $($init:expr),* , $last:expr   ) => (
    // 	    let b = bitset!( $init );
    // 	    b.insert( $last );
    // 	);
    // }

    #[test]
    /// Check cover counts using additive noise for empty model
    fn test_empty_cover() {
	let universe = vec!( 0, 1, 2, 3, 4, 5 );
	// item 0 never occurs
	let data: Vec<Transaction> = vec!(
	    vec!( 1 ),
	    vec!( 1, 2 ),
	    vec!( 1, 2, 3 ),
	    vec!( 1, 2, 3, 4 ),
	    vec!( 1, 2, 3, 4, 5 )
	).iter().map( |t| vector_to_bitset( t ) )
	    .collect();

	let model = BernoulliAssignment::new( &universe );
	let cover: UseCount = model.cover( data.iter().map( |t| (t.clone(), 1) ) ); // make a vector of transaction a data base for testing

	assert_eq!( cover.number_transactions, 5 );
	let add_count_expect = vec!( 0, 5, 4, 3, 2, 1 );
	let kill_count_expect = vec!( 0, 0, 0, 0, 0, 0 );
	for item in 0 .. universe.len() {
	    assert_eq!( cover.get_item_cover_count( item, true, false ), add_count_expect[ item ] );
	    assert_eq!( cover.get_item_cover_count( item, false, true ), kill_count_expect[ item ] );
	}
    }

    #[test]
    /// Check fitted noise parameters for empty model
    fn test_empty_fit() {
	let universe = vec!( 0, 1, 2, 3 );
	let transactions: Vec<Vec<usize>> = vec!(
	    vec!( 1, 2, 3 ),
	    vec!( 1, 2 ),
	    vec!( 1, 3 ),
	    vec!( 1, 2 ),
	);
	let transactions: Vec<Transaction> = transactions.iter()
	    .map( |t: &Vec<usize>| BitSet::from_iter( t.iter().map( |i| *i) ))
	    .collect();

	let mut cover = UseCount::new( universe.clone() );
	for t in &transactions {
	    cover.add_cover( t, &[], 1 );
	}

	let mut model = BernoulliAssignment::new( &universe );
	model.fit( &cover );

	let mut param = Parameters::new();
	/* fit for data
	   1 2 3
	   1 2
	   1 3
	   1 2
	*/
	param.set_number_transactions( 4 );
	param.set_additive_bias( 1, 2 );
	param.set_destructive_bias( 1, 2 );
	param.fit_additive_noise_map(0, 0, 4);
	param.fit_additive_noise_map(1, 4, 0);
	param.fit_additive_noise_map(2, 3, 1);
	param.fit_additive_noise_map(3, 2, 2);

	let denom = 4.0 + 3.0;
	assert_approx!( param.get_additive_noise( 0 ), 1.0 / denom, 0.01 );
	assert_approx!( param.get_additive_noise( 1 ), 5.0 / denom, 0.01 );
	assert_approx!( param.get_additive_noise( 2 ), 4.0 / denom, 0.01 );
	assert_approx!( param.get_additive_noise( 3 ), 3.0 / denom, 0.01 );
	// as representative for all the others
	assert_approx!( param.get_destructive_noise( 0 ), 1.0 / 3.0, 0.01 );
    }

    #[test]
    /// Check log likelihood for empty model.
    fn test_empty_loglik() {
	let universe = vec!( 0, 1, 2, 3 );
	let transactions: Vec<Vec<usize>> = vec!(
	    vec!( 1, 2, 3 ),
	    vec!( 1, 2 ),
	    vec!( 1, 3 ),
	    vec!( 1, 2 ),
	);
	let transactions: Vec<Transaction> = transactions.iter()
	    .map( |t: &Vec<usize>| BitSet::from_iter( t.iter().map( |i| *i) ))
	    .collect();

	let mut cover = UseCount::new( universe.clone() );
	for t in &transactions {
	    cover.add_cover( t, &[], 1 );
	}

	let mut model = BernoulliAssignment::new( &universe );
	// can it handle no zero noise items?
	model.parameters.set_number_transactions( 4 );
	model.parameters.set_additive_noise( 0, 0.0 );
	model.parameters.set_additive_noise( 1, 0.6 );
	model.parameters.set_additive_noise( 2, 0.4 );
	model.parameters.set_additive_noise( 3, 0.2 );
	model.parameters.set_destructive_noise( 0, 0.0 );
	model.parameters.set_destructive_noise( 1, 0.2 );
	model.parameters.set_destructive_noise( 2, 0.2 );
	model.parameters.set_destructive_noise( 3, 0.8 );

	let expected =
	    // item 0
	    0.0 + 4.0 * f64::log2( 1.0 ) + 0.0 + 0.0 * f64::log2( 1.0 ) + 
	    // item 1
	    4.0 * f64::log2( 0.6 ) + 0.0 * f64::log2( 0.4 ) + 0.0 * f64::log2( 0.2 ) + 0.0 * f64::log2( 0.8 ) +
	    // item 2
	    3.0 * f64::log2( 0.4 ) + 1.0 * f64::log2( 0.6 ) + 0.0 * f64::log2( 0.2 ) + 0.0 * f64::log2( 0.8 ) +
	    // item 3
	    2.0 * f64::log2( 0.2 ) + 2.0 * f64::log2( 0.8 ) + 0.0 * f64::log2( 0.8 ) + 0.0 * f64::log2( 0.2 );
	let calculated: f64 = model.calc_loglik( &cover );
	assert_approx!( expected, calculated, 0.01 );
    }

    #[test]
    /// Check if the log likelihood blows up for an impossible covering
    fn test_item_loglik_blowup() {
	let universe = vec!( 0, 1 );
	let transaction = vec!( 0 );
	let transaction = vector_to_bitset( &transaction );

	// initialize model to extreme values
	let mut model = BernoulliAssignment::new( &universe );
	model.parameters.set_additive_noise( 0, 0.0 );
	model.parameters.set_destructive_noise( 0, 0.0 );
	model.parameters.set_additive_noise( 1, 1.0 );
	model.parameters.set_destructive_noise( 1, 1.0 );

	let mut cover = UseCount::new( universe.clone() );
	cover.add_cover( &transaction, &[], 1 );

	// add noise prob for 0 is 0.0 but is used
	// destructive noise prob for 1 is 1.0 but is not used
	let expected = f64::NEG_INFINITY;
	let calculated = model.calc_loglik( &cover );
	assert_eq!( expected, calculated );
    }

    #[test]
    /// Check the probability of a covering with patterns
    fn test_pattern_loglik() {
	let universe = vec!( 0, 1, 2, 3 );
	let data = vec!(
	    vec!( 0, 1 ),
	    vec!( 1, 2 ),
	    vec!( 0, 1, 3 ),
	    vec!( 1, 2, 3 ),
	    vec!( 0, 1, 2, 3 ),
	    vec!( 0, 2 )
	);
	let data: Vec<Transaction> = data.iter().map( |t| vector_to_bitset( t )).collect();

	let mut model = BernoulliAssignment::new( &universe );
	model.parameters.set_number_transactions( 6 );
	model.parameters.set_additive_noise( 0, 0.5 );
	model.parameters.set_destructive_noise( 0, 0.25 );
	model.parameters.set_additive_noise( 1, 0.75 );
	model.parameters.set_destructive_noise( 1, 0.75 );
	model.parameters.set_additive_noise( 2, 0.25 );
	model.parameters.set_destructive_noise( 2, 0.5 );
	model.parameters.set_additive_noise( 3, 0.75 );
	model.parameters.set_destructive_noise( 3, 0.5 );
	
	let pattern1 = model.create_pattern( [0, 1].iter().map( |itm| *itm ));
	model.parameters.set_pattern_prob( pattern1, 0.25 );
	let pattern2 = model.create_pattern( [1, 2].iter().map( |itm| *itm ));
	model.parameters.set_pattern_prob( pattern2, 0.75 );
	
	let pattern_covers = vec!(
	    vec!( pattern1 ),
	    vec!( pattern2 ),
	    vec!( pattern1 ),
	    vec!( pattern2 ),
	    vec!( pattern1, pattern2 ),
	    vec!( pattern1 )
	);

	let mut cover = UseCount::new( universe.clone() );
	for (t, patterns) in data.iter().zip( pattern_covers.iter() ) {
	    let patterns: Vec<PatternPair> = patterns.iter().map( |token| (*token, model.get_pattern( *token ).unwrap()) ).collect();
	    cover.add_cover( t, &patterns, 1 );
	}

	let delta = 0.01;
	// item 0:  +0.5  -0.25
	let expected = 0.0 * 0.5 + 2.0 * f64::log2( 0.5 );
	let calculated = model.calc_additive_noise_loglik( 0, &cover );
	assert_approx!( calculated, expected, delta );
	let expected = 0.0 * f64::log2( 0.25 ) + 4.0 * f64::log2( 0.75 );
	let calculated = model.calc_destructive_noise_loglik( 0, &cover );
	assert_approx!( calculated, expected, delta );
	// item 1  +0.75  -0.75
	let expected = 0.0 * f64::log2( 0.75 ) + 0.0 * f64::log2( 0.25 );
	let calculated = model.calc_additive_noise_loglik( 1, &cover );
	assert_approx!( calculated, expected, delta );
	let expected = 1.0 * f64::log2( 0.75 ) + 5.0 * f64::log2( 0.25 );
	let calculated = model.calc_destructive_noise_loglik( 1, &cover );
	assert_approx!( calculated, expected, delta );
	// item 2  +0.25  -0.5
	let expected = 1.0 * f64::log2( 0.25 ) + 2.0 * f64::log2( 0.75 );
	let calculated = model.calc_additive_noise_loglik( 2, &cover );
	assert_approx!( calculated, expected, delta );
	let expected =  0.0 * f64::log2( 0.5 ) + 3.0 * f64::log2( 0.5 );
	let calculated = model.calc_destructive_noise_loglik( 2, &cover );
	assert_approx!( calculated, expected, delta );
	// item 3  +0.75  -0.5
	let expected = 3.0 * f64::log2( 0.75 ) + 3.0 * f64::log2( 0.25 );
	let calculated = model.calc_additive_noise_loglik( 3, &cover );
	assert_approx!( calculated, expected, delta );
	let expected = 0.0 * f64::log2( 0.5 ) + 0.0 * f64::log2( 0.5 );
	let calculated = model.calc_destructive_noise_loglik( 3, &cover );
	assert_approx!( calculated, expected, delta );
	// pattern 1
	let expected = 4.0 * f64::log2( 0.25 ) + 2.0 * f64::log2( 0.75 );
	let calculated = model.calc_pattern_loglik( pattern1, &cover );
	assert_approx!( calculated, expected, delta );
	// pattern 2
	let expected = 3.0 * f64::log2( 0.75 ) + 3.0 * f64::log2( 0.25 );
	let calculated = model.calc_pattern_loglik( pattern2, &cover );
	assert_approx!( calculated, expected, delta );
    }

    #[test]
    /// Test if parameters are set correctly when patterns are used.
    fn test_pattern_fit() {
	let universe = vec!( 0, 1, 2 );
	let mut model = BernoulliAssignment::new( &universe );
	model.parameters.set_additive_bias( 0, 0 );
	model.parameters.set_destructive_bias( 0, 0 );
	let pattern1 = model.create_pattern( [0, 2].iter().map( |itm| *itm ));
	let pattern2 = model.create_pattern( [0, 1].iter().map( |itm| *itm ));

	let data = vec!(
	    vec!( 0, 2 ),
	    vec!( 0, 1, 2 ),
	    vec!( 0, 1 ),
	    vec!( 0, 1 ),
	);

	let covering = vec!(
	    vec!( pattern1 ),
	    vec!( pattern1 ),
	    vec!( pattern1 ),
	    vec!(),
	);

	let mut cover = UseCount::new( universe.clone() );
	for (t, cov) in data.iter().zip( covering.iter() ) {
	    let t = vector_to_bitset( t );
	    let cov: Vec<PatternPair> = cov.iter().map( |token| (*token, model.get_pattern( *token ).unwrap() )).collect();
	    cover.add_cover( &t, &cov, 1 );
	}

	model.fit( &cover );
	println!( "cover {cover:?}" );
	println!( "parameters {:?}", model.parameters );
	// item 0
	assert_approx!( model.parameters.get_additive_noise( 0 ), 1.0 / 1.0, 0.01 );
	assert_approx!( model.parameters.get_destructive_noise( 0 ), 0.0 / 3.0, 0.01 );
	// item 1
	assert_approx!( model.parameters.get_additive_noise( 1 ), 3.0 / 4.0, 0.01 );
	assert_approx!( model.parameters.get_destructive_noise( 1 ), 0.0, 0.01);
	// item 2
	assert_approx!( model.parameters.get_additive_noise( 2 ), 0.0, 0.01 );
	assert_approx!( model.parameters.get_destructive_noise( 2 ), 1.0 / 3.0, 0.01 );
	// patterns
	assert_approx!( model.parameters.get_pattern_prob( pattern1 ), 3.0 / 4.0, 0.01 );
	assert_approx!( model.parameters.get_pattern_prob( pattern2 ), 0.0, 0.01 );
    }

    #[test]
    /// Check if transactions are covered correctly with patterns.
    fn test_pattern_greedy_cover() {
	let universe = vec!( 0, 1, 2 );
	let mut model = BernoulliAssignment::new( &universe );
	let pat1 = model.create_pattern( [0, 1].iter().map( |itm| *itm ));
	let pat2 = model.create_pattern( [0, 2].iter().map( |itm| *itm ));
	// set parameters s.t. pat1 + noise is used over pat1 and pat2 
	model.parameters.set_additive_noise( 0, 0.01 ); // don't explain as noise
	model.parameters.set_destructive_noise( 0, 0.8 ); // ok if it's missing
	model.parameters.set_additive_noise( 1, 0.05 );
	model.parameters.set_destructive_noise( 1, 0.25 );
	model.parameters.set_additive_noise( 2, 0.25 ); // more likely to use it additively than 1
	model.parameters.set_destructive_noise( 2, 0.25 ); // less likely to use it destructively than 1
	model.parameters.set_pattern_prob( pat1, 0.4 ); // very likely
	model.parameters.set_pattern_prob( pat2, 0.3 ); // not so likely, better to use noise if overlapping with pat1

	evaluate_cover( &model, vec!( 0, 1 ), vec!( pat1 ), vec!( pat2 ) );
	evaluate_cover( &model, vec!( 0, 2 ), vec!( pat2 ), vec!( pat1 ) );
	evaluate_cover( &model, vec!( 1, 2 ), vec!( pat1 ), vec!( pat2 ) );
	evaluate_cover( &model, vec!( 0, 1, 2 ), vec!( pat1, pat2 ), vec!() );
    }

    fn evaluate_cover( model: &BernoulliAssignment, transaction: Vec<Item>, expect_in: Vec<Token>, expect_out: Vec<Token> ) {
	let t = vector_to_bitset( &transaction );
	let cover: HashSet<Token> = model.cover_transaction_greedy( &t ).iter().map( |(token, _)| *token ).collect();
	for pat in expect_in {
	    assert!( cover.contains( &pat ),  "expect {} for {:?}", pat, transaction );
	}
	for pat in expect_out {
	    assert!( !cover.contains( &pat ), "not expect {} for {:?}", pat, transaction );
	}
    }

    #[test]
    /// Check if candidates are generated correctly and in the right order.
    fn test_candidate_generation() {
	let universe = vec!( 0, 1, 2 );
	let mut patterns: Vec<InternalPattern> = Vec::new();
	patterns.push( vec!( 0, 1 ));
	patterns.push( vec!( 1, 2 ));
	for item in &universe {
	    patterns.push( vec!( *item ));
	}

	// create a data set where [0, 1] dominates over [1, 2]
	// total: 10 transactions
	let data = vec!(
	    ( vec!( 0 ), 0),
	    ( vec!( 1 ), 3),
	    ( vec!( 2 ), 0),
	    ( vec!( 0, 1 ), 0),
	    ( vec!( 0, 2 ), 1),
	    ( vec!( 1, 2 ), 1),
	    ( vec!( 0, 1, 2 ), 5 )
	);
	let database = LinkedTrieDatabaseBuilder::new( 3 );
	let mut database = database.build_with_edgelist();
	for (transaction, count) in &data {
	    for i in 0 .. *count {
		database.add( [transaction].iter().copied() );
	    }
	}

	let mut parameters = Parameters::new(); // we only need the additive noise parameters
	parameters.set_number_transactions( 7 );
	parameters.fit_additive_noise_map( 0, 6, 4 );
	parameters.fit_additive_noise_map( 1, 9, 1 );
	parameters.fit_additive_noise_map( 2, 7, 3 );

	let expected_candidates = vec!( vec!( 0, 1, 2 ), vec!( 0, 2 ) );
	let mut generator = PatternRecombinator::new( patterns.iter().cloned().collect(), &parameters );
	generator.combine_patterns( &patterns, &database );
	generator.sort_candidates();
	let calculated_candidates: Vec<CandidatePattern> = generator.collect();
	println!( "{:?}", calculated_candidates );
	assert_eq!( expected_candidates.len(), calculated_candidates.len() );
	for (expected, calculated) in expected_candidates.iter().zip( calculated_candidates.iter() ) {
	    println!( "Expected {:?} got {:?}", expected, calculated );
	    // candidate patterns contain items in order
	    assert_eq!( expected, &calculated.pattern );
	}
    }

    #[test]
    /// Check if parameter fit is correct with prior
    fn test_bias_fit() {
	let mut param = Parameters::new();
	let (noise_plus_bias, noise_min_bias, noise_total_bias) = (1, 2, 3);
	let (pattern_plus_bias, pattern_min_bias, pattern_total_bias) = (1, 1, 2);
	param.set_number_transactions( 6 );
	param.set_additive_bias( noise_plus_bias, noise_min_bias );
	param.set_destructive_bias( noise_plus_bias, noise_min_bias );
	param.set_pattern_bias( pattern_plus_bias, pattern_min_bias );

	param.fit_additive_noise_map( 0, 2, 1 ); // 2/3
	param.fit_destructive_noise_map( 0, 1, 2  ); // 1/3
	param.fit_additive_noise_map( 1, 6, 0 ); // 6/6
	param.fit_destructive_noise_map( 1, 0, 0); // 0/0p
	param.fit_pattern_prob_map( 0, 5, 1 ); // 5/6
	
	assert_approx!( param.get_additive_noise( 0 ), (2 + noise_plus_bias) as f64 / (3 + noise_total_bias) as f64, 0.01 );
	assert_approx!( param.get_destructive_noise( 0 ), (1 + noise_plus_bias) as f64 / (3 + noise_total_bias) as f64, 0.01 );
	assert_approx!( param.get_additive_noise( 1 ), (6 + noise_plus_bias) as f64 / (6 + noise_total_bias) as f64, 0.01 );
	assert_approx!( param.get_destructive_noise( 1 ), (0 + noise_plus_bias) as f64 / (0 + noise_total_bias) as f64, 0.01 );
	assert_approx!( param.get_pattern_prob( 0 ), (5 + pattern_plus_bias) as f64 / (6 + pattern_total_bias) as f64, 0.01 );
    }

    #[test]
    /// Test the joint distributions for counts and parameters
    fn test_log_prior() {
	let mut param = Parameters::new();
	let (noise_plus_bias, noise_min_bias) = (1, 2);
	let (pattern_plus_bias, pattern_min_bias) = (1, 1);
	param.set_additive_bias( noise_plus_bias, noise_min_bias );
	param.set_destructive_bias( noise_plus_bias, noise_min_bias );
	param.set_pattern_bias( pattern_plus_bias, pattern_min_bias );

	let additive_noise_0 = 0.3;
	let destructive_noise_0 = 0.001;
	let additive_noise_1 = 0.001;
	let destructive_noise_1 = 0.999;
	let pattern_prob = 0.75;
	param.set_additive_noise( 0, additive_noise_0 );
	param.set_destructive_noise( 0, destructive_noise_0 );
	param.set_additive_noise( 1, additive_noise_1 );
	param.set_destructive_noise( 1, destructive_noise_1 );
	param.set_pattern_prob( 0, pattern_prob );

	assert_approx!( param.calc_additive_logprior( 0 ), f64::log2( 1.764 ), 0.01 );
	assert_approx!( param.calc_destructive_logprior( 0 ), f64::log2( 0.012 ), 0.01 );
	assert_approx!( param.calc_additive_logprior( 1 ), f64::log2( 0.012 ), 0.01 );
	assert_approx!( param.calc_destructive_logprior( 1 ), f64::log2( 0.000012 ), 0.01 );
	assert_approx!( param.calc_pattern_logprior( 0 ), f64::log2( 1.125 ), 0.01 );
    }

    fn vector_to_bitset( items: &Vec<usize> ) -> BitSet {
	let mut set = BitSet::new();
	for item in items {
	    set.insert( *item );
	}
	set
    }
}
