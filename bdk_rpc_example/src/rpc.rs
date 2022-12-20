use std::{
    collections::BTreeMap,
    mem,
    sync::mpsc::{SendError, SyncSender},
};

use bdk_core::bitcoin::{Block, BlockHash, Transaction};
use bitcoincore_rpc::{Auth, Client as RpcClient, RpcApi};

pub enum RpcData {
    Start {
        starting_tip: u32,
        target_tip: u32,
    },
    Blocks {
        // the height of the first block in the vector
        first_height: u32,
        blocks: Vec<Block>,
    },
    Mempool(Vec<Transaction>),
    BlocksSynced,
    MempoolSynced,
}

#[derive(Debug)]
pub enum RpcError {
    Rpc(bitcoincore_rpc::Error),
    Send(SendError<RpcData>),
}

impl From<bitcoincore_rpc::Error> for RpcError {
    fn from(err: bitcoincore_rpc::Error) -> Self {
        Self::Rpc(err)
    }
}

impl From<SendError<RpcData>> for RpcError {
    fn from(err: SendError<RpcData>) -> Self {
        Self::Send(err)
    }
}

impl core::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for RpcError {}

pub struct Client {
    client: RpcClient,
}

impl Client {
    pub fn new(url: &str, auth: Auth) -> Result<Self, RpcError> {
        let client = RpcClient::new(url, auth)?;
        Ok(Client { client })
    }

    pub fn emit_blocks(
        &self,
        chan: SyncSender<RpcData>,
        local_cps: &mut BTreeMap<u32, BlockHash>,
        fallback_height: u32,
        block_batch_size: usize,
    ) -> Result<(), RpcError> {
        let tip = self.client.get_block_count()? as u32;

        let mut start_emitting_at = fallback_height;

        for (height, hash) in local_cps.iter().rev() {
            if self.client.get_block_stats(*height as u64)?.block_hash == *hash {
                start_emitting_at = height + 1;
            }
        }

        chan.send(RpcData::Start {
            target_tip: tip,
            starting_tip: start_emitting_at,
        })?;

        let mut curr_hash = self
            .client
            .get_block_stats(start_emitting_at as u64)?
            .block_hash;
        let mut curr_height = start_emitting_at;
        let mut first_height = curr_height;
        let mut block_buffer = Vec::with_capacity(block_batch_size);

        loop {
            let block_info = self.client.get_block_info(&curr_hash)?;
            let block = self.client.get_block(&curr_hash)?;
            block_buffer.push(block);

            if block_buffer.len() == block_batch_size || block_info.nextblockhash.is_none() {
                let emitted_blocks =
                    mem::replace(&mut block_buffer, Vec::with_capacity(block_batch_size));
                chan.send(RpcData::Blocks {
                    first_height,
                    blocks: emitted_blocks,
                })?;
                first_height = curr_height + 1;
            }

            curr_hash = match block_info.nextblockhash {
                Some(nextblockhash) => nextblockhash,
                None => break,
            };
            curr_height += 1;
        }

        chan.send(RpcData::BlocksSynced)?;

        Ok(())
    }

    pub fn emit_mempool(
        &self,
        chan: SyncSender<RpcData>,
        tx_batch_size: usize,
    ) -> Result<(), RpcError> {
        for txids in self.client.get_raw_mempool()?.chunks(tx_batch_size) {
            let txs = txids
                .iter()
                .map(|txid| self.client.get_raw_transaction(txid, None))
                .collect::<Result<Vec<_>, _>>()?;
            chan.send(RpcData::Mempool(txs))?;
        }

        chan.send(RpcData::MempoolSynced)?;

        Ok(())
    }
}

impl bdk_cli::Broadcast for Client {
    type Error = RpcError;
    fn broadcast(&self, tx: &Transaction) -> bdk_cli::anyhow::Result<(), Self::Error> {
        let _txid = self.client.send_raw_transaction(tx)?;
        Ok(())
    }
}
