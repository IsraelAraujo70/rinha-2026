use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Read},
    path::PathBuf,
};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fraud::build::build_index_from_json_reader;

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let input = args
        .next()
        .map(PathBuf::from)
        .context("usage: build_index <references.json.gz> <data.bin>")?;
    let output = args
        .next()
        .map(PathBuf::from)
        .context("usage: build_index <references.json.gz> <data.bin>")?;

    let input_file = File::open(&input).with_context(|| format!("open {}", input.display()))?;
    let reader: Box<dyn Read> = if input.extension().is_some_and(|ext| ext == "gz") {
        Box::new(GzDecoder::new(BufReader::new(input_file)))
    } else {
        Box::new(BufReader::new(input_file))
    };

    let output_file =
        File::create(&output).with_context(|| format!("create {}", output.display()))?;
    let writer = BufWriter::new(output_file);
    let count = build_index_from_json_reader(reader, writer)?;
    eprintln!("wrote {count} records to {}", output.display());
    Ok(())
}
