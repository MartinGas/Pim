
use crate::*;

mod bernoulli;

/// Token to represent a pattern
pub type Token = usize;

pub trait Model {
    /// contains sufficient stats gathered from the data set
    type Cover;
    type Candidate;

    /// Covers the data with patterns to obtain the model's stats
    fn cover <I> ( &self, data: I ) -> Self::Cover where I: Iterator<Item = DataPair>;

    /// Fits the model parameters to the cover and returns the log likelihood
    fn fit( &mut self, cover: &Self::Cover );

    /// Calculates the log likelihood of the given cover under this model
    fn calc_loglik( &self, cover: &Self::Cover ) -> f64;

    /// Generates candidates to better explain the provided data base.
    fn generate_candidates <D: Database> ( &self, data: &D ) -> Box<dyn Iterator<Item = Self::Candidate>>;

    /// Adds a candidate to the model
    fn add_candidate( &mut self, candidate: &mut Self::Candidate );

    /// Removes a candidate from the model
    fn remove_candidate( &mut self, candidate: &mut Self::Candidate );
}

pub use bernoulli::{BernoulliAssignment, BernoulliFormatter};
