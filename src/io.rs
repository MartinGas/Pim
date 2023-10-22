use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};

use bit_set::BitSet;

use crate::{Transaction, Itemvec, Item};

pub type DataGenerator<T> = Box<dyn Iterator<Item = T>>;

/// Reads data in FIMI format into a data base. Creates data using the converter, which is given a line
pub fn read_data<T, F>( path: &str, converter: F ) -> Result<DataGenerator<T>, String> where
    F: Fn(&str) -> Option<T>,
{
    let path = Path::new( path );
    let file = File::open( path ).map_err( |e| e.to_string() )?;
    let reader = BufReader::new( file );
    let generator = reader.lines()
        .filter_map( |l| l.ok() )
	// \todo set reasonable initial capacity for faster reading
        .filter_map( |l| converter( &l ));
    Result::Ok( Box::new( generator ))
}

/// Parses numbers seprated by splitter into vector
pub fn parse_fimi_to_vec( line: &str, splitter: &str ) -> Option<Itemvec> {
    let mut items = Itemvec::new();
    for chunk in line.split( splitter ) {
	match Item::from_str_radix( chunk, 10 ) {
	    Ok( item ) => items.push( item ),
	    Err( _ ) => return None,
	}
    }
    Some( items )
}

/// Parses numbers separated by splitter into bitset
pub fn parse_fimi_to_bitset( line: &str, splitter: &str, capacity: usize ) -> Option<BitSet> {
    let mut transaction = Transaction::with_capacity( capacity );
    for chunk in line.split( splitter ) {
	match usize::from_str_radix( chunk, 10 ) {
	    Ok( item ) => { transaction.insert( item ); },
	    Err( _ ) => return None,
	}
    }
    Some( transaction )
}
