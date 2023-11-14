
use serde;

use crate::*;
use crate::io::{PrettyFormatter, produce_fimi};

use super::{BernoulliAssignment, Token};

pub struct BernoulliFormatter {
    show_items: bool,
    show_patterns: bool,
}

impl PrettyFormatter<BernoulliAssignment> for BernoulliFormatter {

    fn format_pretty( &self, model: &BernoulliAssignment ) -> String {
	let mut output = String::new();
	output.push( '\n' ); // so output begins on a new line
	
	let mut ordered_items: Itemvec = model.iterate_universe().collect();
	if self.show_items {
	    ordered_items.sort();
	    output = ordered_items.iter()
		.map( |item| {
		    let parameters = model.get_parameters();
		    let add_noise = parameters.get_additive_noise( *item );
		    let kill_noise = parameters.get_destructive_noise( *item );
		    format_item( *item, add_noise, kill_noise )
		}).fold( output, |acc, item_string| join_lines( acc, item_string ));
	}

	if self.show_patterns {
	    let mut ordered_tokens: Vec<Token> = model.iterate_tokens().collect();
	    ordered_tokens.sort();
	    output = ordered_tokens.iter()
		.map( |token| {
		    let pattern = model.get_pattern( *token ).expect( "tokens in model have pattern" );
		    let prob = model.get_parameters().get_pattern_prob( *token );
		    format_pattern( *token, pattern, prob )
		}).fold( output, |acc, pattern_string| join_lines( acc, pattern_string ));
	}
	output
    }
}

impl serde::Serialize for BernoulliAssignment {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
	self.patterns.serialize( serializer )
    }
}

fn format_item( item: Item, additive_noise: f64, destructive_noise: f64 ) -> String {
    format!( "{item}:  +{additive_noise:.3}  -{destructive_noise:.3}" )
}

fn format_pattern( token: Token, items: &Itemvec, probability: f64 ) -> String {
    format!( "{token}:  {probability:.3} {}", produce_fimi( items.iter().copied(), "", " ", "" ))
}

fn join_lines( mut accumulator: String, addition: String ) -> String {
    accumulator.push_str( addition.as_str() );
    accumulator.push( '\n' );
    accumulator
}

impl BernoulliFormatter {
    pub fn new() -> BernoulliFormatter {
	BernoulliFormatter{
	    show_items: false,
	    show_patterns: false,
	}
    }

    pub fn show_items( &mut self ) { self.show_items = true; }
    pub fn show_patterns( &mut self ) { self.show_patterns = true; }
}
