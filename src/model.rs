
use crate::{Item, Transaction, Count, Database};

mod bernoulli;

/// Token to represent a pattern
pub type Token = usize;

pub trait Model {
    /// contains sufficient stats gathered from the data set
    type Cover;
    type Candidate;

    /// Covers the data with patterns to obtain the model's stats
    fn cover <'a, I> ( &self, data: I ) -> Self::Cover where I: Iterator<Item = (&'a Transaction, Count)>;

    /// Fits the model parameters to the cover and returns the log likelihood
    fn fit( &mut self, cover: &Self::Cover );

    /// Calculates the log likelihood of the given cover under this model
    fn calc_loglik( &self, cover: &Self::Cover ) -> f64;

    /// Generates candidates to better explain the provided data base.
    /// The lifetime of the generator is tied to the lifetime of the model.
    fn generate_candidates <'a, 'b, D: Database> ( &'a self, data: &'b D ) -> Box<dyn Iterator<Item = Self::Candidate> + 'a>;
}

