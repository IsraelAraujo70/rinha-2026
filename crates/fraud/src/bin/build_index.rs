use std::{
    env,
    fs::File,
    io::{BufReader, BufWriter, Seek, SeekFrom, Write},
    path::PathBuf,
};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use fraud::{
    index::{encode_record, write_header},
    vector::Vector,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vector,
    label: String,
}

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
    let decoder = GzDecoder::new(BufReader::new(input_file));
    let records: Vec<ReferenceRecord> =
        serde_json::from_reader(decoder).with_context(|| format!("parse {}", input.display()))?;

    let output_file =
        File::create(&output).with_context(|| format!("create {}", output.display()))?;
    let mut writer = BufWriter::new(output_file);
    write_header(&mut writer, 0)?;

    let mut count = 0u64;
    for record in &records {
        writer.write_all(&encode_record(&record.vector, &record.label)?)?;
        count += 1;
    }

    writer.flush()?;
    let mut output_file = writer.into_inner()?;
    output_file.seek(SeekFrom::Start(0))?;
    write_header(&mut output_file, count)?;
    output_file.flush()?;
    eprintln!("wrote {count} records to {}", output.display());
    Ok(())
}
