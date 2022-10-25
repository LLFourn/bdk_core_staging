use core::array::TryFromSliceError;
#[cfg(feature = "std")]
use std::convert::TryFrom;
#[cfg(feature = "std")]
use std::{ fmt, error::Error };
#[cfg(feature = "std")]
use std::io::{self, Read, Seek, Write};

use bitcoin::{consensus::{self, encode}, BlockHash, Txid, Transaction, TxOut};

use crate::{collections::*, Vec, tx_graph::{TxGraph, TxNode}, sparse_chain::{SparseChain, TxHeight, UpdateFailure } };

/// Persistence error of [`SparseChain`] and [`TxGraph`]
#[derive(Debug)]
pub enum IOError {
    /// wrapper for std::io::Error
    Io(io::Error),
    /// Value used for kind_byte is unknown
    UnknownKind(u8),
    /// consensus deserialize/serialize error
    Consensus(consensus::encode::Error),
    /// slice to array conversion error
    SliceConversion(TryFromSliceError),
    /// [`TxGraph`] update error
    TxUpdateError(UpdateFailure)
}

#[cfg(feature = "std")]
impl Error for IOError {}

impl fmt::Display for IOError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<io::Error> for IOError {
    fn from(error: std::io::Error) -> IOError {
        IOError::Io(error)
    }
}

impl From<encode::Error> for IOError {
    fn from(error: encode::Error) -> IOError {
        IOError::Consensus(error)
    }
}

impl From<TryFromSliceError> for IOError {
    fn from(error: TryFromSliceError) -> IOError {
        IOError::SliceConversion(error)
    }
}

impl From<UpdateFailure> for IOError {
    fn from(error: UpdateFailure) -> IOError {
        IOError::TxUpdateError(error)
    }
}

const UNCONFIRMED_TX_HEIGHT: u32 = u32::MAX - 1;

/// write SparseChain to disk for persistence
pub fn write_sparse_chain<T: Write + Seek + Read>(chain: &SparseChain, writer: T) -> Result<(), IOError> {
    let bytes = writer.bytes().collect::<io::Result<Vec<u8>>>()?;
    let mut cursor = io::Cursor::new(bytes);
    // Todo: check if the state_id of the in-memory store is different from disk store
    // if that is the case, write new data to disk, if not return because there is
    // nothing to write.

    // update version
    // TODO: write current db version
    let _size = cursor.write(&[0u8; 4])?;

    // update state_id
    // TODO: write current state_id
    let _size = cursor.write(&[0u8; 4])?;

    
    // TODO: Figure out what to do if the checkpoint is None
    if let Some(checkpoint_limit) = chain.checkpoint_limit {
        let _size = cursor.write(&u32::to_be_bytes(checkpoint_limit as u32))?;

        let mut iter_checkpoints = chain.checkpoints.iter();
        for _i in 0..checkpoint_limit {
            if let Some((height, block_hash)) = iter_checkpoints.next() {
                let mut buf = [0u8; 36];
                u32::to_be_bytes(*height).iter().enumerate().for_each(|(i, val)| buf[i] = *val);
                encode::serialize(block_hash).iter().enumerate().for_each(|(i, val)| buf[i + 4] = *val);
                let _size = cursor.write(&buf)?;
            } else {
                let zero_pad_buf = [0u8; 36];
                let _size = cursor.write(&zero_pad_buf)?;
            }
        }
    }

    for (height, txid) in &chain.txid_by_height {
        let mut buf = match height {
            TxHeight::Confirmed(block_height)=> u32::to_be_bytes(*block_height).to_vec(),
            TxHeight::Unconfirmed => u32::to_be_bytes(UNCONFIRMED_TX_HEIGHT).to_vec()
        };
        buf.extend(encode::serialize(txid));
        let _size = cursor.write(&buf)?;
    }

    Ok(())
}

/// Instantiate SparseChain from disk bytes.
pub fn instantiate_sparse_chain<T: Read + Seek>(chain: &mut SparseChain, reader: &mut T) -> Result<(), IOError> {
    let bytes = reader.bytes().collect::<io::Result<Vec<u8>>>()?;
    let mut cursor = io::Cursor::new(bytes);
    // skip the state_id + version
    cursor.set_position(8);
    let mut checkpoint_limit_bytes = [0u8; 4];
    let _size = cursor.read(&mut checkpoint_limit_bytes)?;
    let checkpoint_limit = u32::from_be_bytes(checkpoint_limit_bytes);
    let mut checkpoint_bytes = Vec::with_capacity((checkpoint_limit as u32 * 36u32) as usize);
    // read checkpoint_limit * 36 bytes
    let _size = cursor.read(&mut checkpoint_bytes)?;
    // create checkpoints from these bytes.
    for chunk in checkpoint_bytes.chunks(36) {
        // once you meet the padded zeros stop reading
        if chunk.iter().all(|x| *x == 0u8) {
            break;
        }

        let height = u32::from_be_bytes(<[u8; 4]>::try_from(&chunk[0..4])?);
        let block_hash: BlockHash = encode::deserialize(&chunk[4..])?;
        chain.checkpoints.insert(height, block_hash);
    }

    // read (height + txid) data
    let mut txid_bytes = Vec::new();
    // create (height, txid) mapping and the reverse mapping
    let _size = cursor.read_to_end(&mut txid_bytes)?;
    for chunk in txid_bytes.chunks(36) {
        let height = u32::from_be_bytes(<[u8; 4]>::try_from(&chunk[0..4])?);
        let txid: Txid = encode::deserialize(&chunk[4..])?;
        let _set = if height == (UNCONFIRMED_TX_HEIGHT) {
            chain.insert_tx(txid.clone(), TxHeight::Unconfirmed)?
        } else {
            chain.insert_tx(txid.clone(), TxHeight::Confirmed(height))?
        };
    }
    Ok(())
}

/// write TxGraph to disk for persistence storage
pub fn write_tx_graph<T: Write + Seek>(
    tx_graph: &TxGraph,
    writer: &mut T
) -> io::Result<()> {
    // TODO: write current state_id + version
    let header_bytes = [0u8; 8];
    let _size = writer.write(&header_bytes)?;
    let mut buf = Vec::new();
    for (txid, tx_node) in &tx_graph.txs {
        match tx_node {
            TxNode::Whole(tx) => {
                let serialized_tx = encode::serialize(tx);
                // length of the transaction data
                buf.extend((serialized_tx.len() as u32).to_be_bytes());
                // 1u8 stands for the kind of transaction (Whole).
                buf.push(1u8);
                // add the transaction data
                buf.extend(serialized_tx);
            }
            TxNode::Partial(outpoints) => {
                buf.extend(((outpoints.len()) as u32).to_be_bytes());
                buf.push(2u8);
                let serialized_txid = encode::serialize(txid);
                // add txid
                buf.extend(serialized_txid);
                for (idx, output) in outpoints {
                    let serialized_txout = encode::serialize(output);
                    let idx = (*idx as u32).to_be_bytes();
                    // add length of new item (idx + output)
                    buf.extend(((idx.len() + serialized_txout.len()) as u32).to_be_bytes());
                    // add index
                    buf.extend(idx);
                    // add output
                    buf.extend(serialized_txout);
                }
            }
        }
    }
    let _size = writer.write(&buf)?;
    Ok(())
}

/// Instantiate TxGraph from disk byte data.
pub fn instantiate_tx_graph<T: Read + Seek>(
    tx_graph: &mut TxGraph,
    reader: &mut T,
) -> Result<(), IOError> {
    let bytes = reader.bytes().collect::<io::Result<Vec<u8>>>()?;
    let mut cursor = io::Cursor::new(bytes);

    // skip the state_id + version bytes
    // TODO: read version + state_id
    cursor.set_position(8);

    //while we haven't reached the end of the bytes iterator
    // we want to read the first/length byte, read the type byte,
    let mut length_bytes = [0u8; 4];
    while cursor.read(&mut length_bytes)? != 0 {
        let mut kind_byte = [0u8];
        let _size = cursor.read(&mut kind_byte)?;

        if kind_byte[0] == 1 {
            // if the kind byte is full, read the next (length amount of) bytes,
            // consensus deserialize that to full transaction
            let length = u32::from_be_bytes(length_bytes);
            //let mut tx_data = [0u8, length];
            let mut tx_data = Vec::with_capacity(length as usize);
            let _size = cursor.read(&mut tx_data)?;
            let tx: Transaction = encode::deserialize(&tx_data)?;
            let _inserted_tx = tx_graph.txs.insert(tx.txid(), TxNode::Whole(tx));
        } else if kind_byte[0] == 2 {
            // type is partial, read all bytes to create partial transaction.
            let mut partial_tx: BTreeMap<u32, TxOut> = BTreeMap::new();
            //get length of map
            let map_length = u32::from_be_bytes(length_bytes);
            let mut txid_bytes = [0u8; 4];
            let _size = cursor.read(&mut txid_bytes)?;
            for _i in 0..map_length {
                //get the size of the item
                let mut item_size = [0u8; 4];
                let _size = cursor.read(&mut item_size)?;
                //read size element of items
                let item_size = u32::from_be_bytes(item_size);
                let mut data = Vec::with_capacity(item_size as usize);
                let _size = cursor.read(&mut data)?;
                //separate idx from output.
                let idx = u32::from_be_bytes(<[u8; 4]>::try_from(&data[4..8])?);
                let output = encode::deserialize(&data[8..])?;
                //deserialize and add to map.
                partial_tx.insert(idx, output);
            }
            let txid = encode::deserialize(&txid_bytes)?;
            let _inserted_tx = tx_graph.txs.insert(txid, TxNode::Partial(partial_tx));
        } else {
            return Err(IOError::UnknownKind(kind_byte[0]));
        }
    }
    Ok(())
}