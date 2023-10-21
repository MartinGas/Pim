
use std::fmt::Debug;

use crate::*;


pub trait Miner {
    fn mine<'a, D, M>( &'a mut self, data: &'a D, model: &'a mut M ) where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Loggable + 'a,
	M::Candidate: Loggable;
}

pub struct EmMiner {
    max_iterations: u64,
}

impl Miner for EmMiner {
    fn mine<'a, D, M>( &'a mut self, data: &'a D, model: &'a mut M ) where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Loggable + 'a,
	M::Candidate: Loggable,
    {
	let mut last_loglik = f64::NEG_INFINITY;
	let mut loglik = self.initialize( model, data );
	let mut iteration = 0;
	while last_loglik < loglik {
	    iteration += 1;
	    let iteration_span = info_span!( "iteration", number = iteration );
	    let last_loglik = loglik;
	    loglik = self.iterate( model, data, loglik );

	    info!( "Likelihood changed from {last_loglik:.3} to {loglik:.3}" );
	}
    }
}



impl EmMiner {
    pub fn new( max_iterations: u64 ) -> EmMiner {
	EmMiner {
	    max_iterations,
	}
    }

    fn initialize<'a, M: Model, D: Database>( &self, model: &mut M, data: &'a D ) -> f64 where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Loggable + 'a,
    {
	let init_span = info_span!( "initialization" );
	let mut loglik = self.step( model, data );

	model.log( "initial model", Level::DEBUG );
	info!( "initial score {}", loglik );

	loglik
    }

    fn iterate<'a, M: Model, D: Database>( &self, model: &mut M, data: &'a D, loglik: f64 ) -> f64 where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Loggable + 'a,
    M::Candidate: Loggable,
    {
	let success = self.grow( model, data, loglik );
	if success {
	    debug!( "Candidate accepted" );
	    
	    self.optimize( model, data, loglik )
	} else {
	    loglik
	}
    }

    fn step<'a, M: Model, D: Database>( &self, model: &mut M, data: &'a D ) -> f64
    where
	&'a D: IntoIterator<Item = DataPair>, D: 'a
    {
	let cover = model.cover( data.into_iter() );
	model.fit( &cover );
	model.calc_loglik( &cover )
    }

    fn optimize<'a, M: Model, D: Database>( &self, model: &mut M, data: &'a D, base_loglik: f64 ) -> f64 where
	&'a D: IntoIterator<Item = DataPair>,
    {
	let mut loglik = base_loglik;
	let mut last_loglik = f64::NEG_INFINITY;

	while loglik > last_loglik {
	    last_loglik = loglik;
	    loglik = self.step( model, data );
	}
	loglik
    }

    fn grow<'a, M: Model, D: Database + 'a>( &self, model: &mut M, data: &'a D, loglik: f64 ) -> bool where
	&'a D: IntoIterator<Item = DataPair>,
	M::Candidate: Loggable,
    {
	let candidates = model.generate_candidates( data );
	for mut current in candidates {
	    current.log( "adding candidate", Level::DEBUG );
	    
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
