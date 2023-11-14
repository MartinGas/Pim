use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead, Write};

use serde_json as json;

use bit_set::BitSet;

use crate::{Transaction, Itemvec, Item};

/// Converts a structure into a string
pub trait PrettyFormatter<T> {
    fn format_pretty( &self, object: &T ) -> String;
}

pub type DataGenerator<T> = Box<dyn Iterator<Item = T>>;

/// Reads data in FIMI format into a data base. Creates data using the converter, which is given a line
pub fn read_data<T, F>( path: &str, converter: F ) -> Result<DataGenerator<T>, String> where
    F: Fn(&str) -> Option<T> + 'static,
{
    let path = Path::new( path );
    let file = File::open( path ).map_err( |e| e.to_string() )?;
    let reader = BufReader::new( file );
    let generator = reader.lines()
        .filter_map( |l| l.ok() )
	// \todo set reasonable initial capacity for faster reading
        .filter_map( move |l| converter( &l ));
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

/// Creates a fimi string from an iterator over items
pub fn produce_fimi<I: Iterator<Item = Item>>( items: I, left_delimiter: &str, separator: &str, right_delimiter: &str ) -> String {
    let add_item_to_string = |mut fimi: String, i: Item| {
	fimi.push_str( i.to_string().as_str() );
	fimi.push_str( separator );
	fimi
    };
    let mut fimi = String::new();
    fimi.push_str( left_delimiter );
    fimi = items.fold( fimi, add_item_to_string );
    fimi.push_str( right_delimiter );
    fimi
}

/// Writes a serializeable model to a file
pub fn write_model<M: serde::Serialize>( model: &M, path: &str ) -> Result<(), String> {
    match serde_json::to_string( model ) {
	json::Result::Ok( model_string ) => {
	    let path = Path::new( path );
	    let mut file = File::create( path ).map_err( |err| err.to_string() )?;
	    write!( file, "{}", model_string ).map_err( |err| err.to_string() )
	},
	json::Result::Err( err ) => return Result::Err( err.to_string() ),
    }
}
