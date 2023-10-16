
use std::fmt::Debug;

use crate::{Database, DataPair, Model};

pub trait Miner {
    fn mine<'a, D, M>( &'a mut self, data: &'a D, model: &'a mut M ) where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Debug + 'a,
	M::Candidate: Debug;
}

pub struct EmMiner {
    max_iterations: u64,
}

impl Miner for EmMiner {
    fn mine<'a, D, M>( &'a mut self, data: &'a D, model: &'a mut M ) where
	D: Database + 'a,
    &'a D: IntoIterator<Item = DataPair>,
	M: Model + Debug + 'a,
	M::Candidate: Debug,
    {
	let mut loglik = self.step( model, data );
	let mut last_loglik = f64::NEG_INFINITY;
	let mut iteration = 0;
	let delta = 0.00001;
	
	while loglik > last_loglik + delta && iteration < self.max_iterations {
	    println!( "Model loglik {loglik:.4}" );
	    println!( "{model:?}" );

	    let success = self.grow( model, data, loglik );
	    if success {
		last_loglik = loglik;
		loglik = self.optimize( model, data, loglik );
		println!( "Success! Improved loglik {last_loglik:.3} -> {loglik:.3}" );
		println!( "{model:?}" );
	    }
	    iteration += 1;
	}
    }
}



impl EmMiner {
    pub fn new( max_iterations: u64 ) -> EmMiner {
	EmMiner {
	    max_iterations,
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
	M::Candidate: Debug,
    {
	let candidates = model.generate_candidates( data );
	for mut current in candidates {
	    model.add_candidate( &mut current );
	    let next_loglik = self.step( model, data );
	    println!( "Added {current:?} yields {next_loglik:?} over {loglik:?}" );
	    if next_loglik > loglik {
		return true;
	    } else {
		model.remove_candidate( &mut current );
	    }
	}
	false
    }


    
    
}
