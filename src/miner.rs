
use std::time::*;

use crate::*;
use crate::io::PrettyFormatter;


pub trait Miner<'a, D, M> {
    fn mine( &'a mut self, data: &'a D, model: &'a mut M );
}

pub struct EmMiner<D, M> {
    max_iterations: u64,
    data: Option<D>,
    model: Option<M>,
    model_formatter: Option<Box<dyn PrettyFormatter<M>>>,
}

impl <'a, D: Database, M: Model> Miner<'a, D, M> for EmMiner<D,M> where
    &'a D: IntoIterator<Item = DataPair> + 'a,
{
    fn mine( &'a mut self, data: &'a D, model: &'a mut M ) where {
	let mut last_loglik = f64::NEG_INFINITY;
	let mut loglik = self.initialize( model, data );
	let mut iteration = 0;
	while last_loglik < loglik {
	    iteration += 1;
	    let iteration_start_time = Instant::now();
	    let _iteration_span = info_span!( "iteration", number = iteration );

	    last_loglik = loglik;
	    loglik = self.iterate( model, data, loglik );

	    let iteration_duration = Instant::now().duration_since( iteration_start_time );
	    info!( "Likelihood changed from {last_loglik:.3} to {loglik:.3} [took {}s]", iteration_duration.as_secs() );
	    if let Some(formatter) = &self.model_formatter {
		debug!( "{}", formatter.format_pretty( model ));
	    }
	}
    }
}



impl <'a, D: Database, M: Model> EmMiner<D, M> where
    &'a D: IntoIterator<Item = DataPair> + 'a,
{
    pub fn new( max_iterations: u64 ) -> EmMiner<D, M> {
	EmMiner {
	    max_iterations,
	    data: None,
	    model: None,
	    model_formatter: None,
	}
    }

    pub fn provide_model_formatter<F: PrettyFormatter<M> + 'static>( &mut self, formatter: F ) {
	self.model_formatter = Some( Box::new( formatter ));
    }

    fn initialize( &self, model: &mut M, data: &'a D ) -> f64 {
	let _init_span = info_span!( "initialization" );
	let loglik = self.step( model, data );

	info!( "initial score {}", loglik );
	// if let Some(formatter) = self.model_formatter {
	    // debug!( "{}", formatter.format_pretty( model ));
	// }

	loglik
    }

    fn iterate( &self, model: &mut M, data: &'a D, loglik: f64 ) -> f64 {
	let success = self.grow( model, data, loglik );
	if success {
	    debug!( "Candidate accepted" );
	    
	    self.optimize( model, data, loglik )
	} else {
	    loglik
	}
    }

    fn step( &self, model: &mut M, data: &'a D ) -> f64
    {
	let cover = model.cover( data.into_iter() );
	model.fit( &cover );
	model.calc_loglik( &cover )
    }

    fn optimize( &self, model: &mut M, data: &'a D, base_loglik: f64 ) -> f64 where
    {
	let mut loglik = base_loglik;
	let mut last_loglik = f64::NEG_INFINITY;

	while loglik > last_loglik {
	    last_loglik = loglik;
	    loglik = self.step( model, data );
	}
	loglik
    }

    fn grow( &self, model: &mut M, data: &'a D, loglik: f64 ) -> bool {
	let candidates = model.generate_candidates( data );
	for mut current in candidates {
	    // current.log( "adding candidate", Level::DEBUG );
	    
	    model.add_candidate( &mut current );
	    let next_loglik = self.step( model, data );

	    debug!( "Candidate yields {:.3} over {:.3}", next_loglik, loglik );

	    if next_loglik > loglik {
		return true;
	    } else {
		model.remove_candidate( &mut current );
	    }
	}
	false
    }


    
    
}
