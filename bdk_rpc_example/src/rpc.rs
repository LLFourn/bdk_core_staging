use std::{
    collections::{BTreeMap, HashSet},
    sync::mpsc::{SendError, SyncSender},
};

use bdk_chain::bitcoin::{Block, BlockHash, Transaction, Txid};
use bitcoincore_rpc::{Auth, Client as RpcClient, RpcApi};

/// Minimum number of transactions to batch together for each emission.
const TX_EMIT_THRESHOLD: usize = 75_000;

pub enum RpcData {
    Start {
        local_tip: u32,
        target_tip: u32,
    },
    Blocks {
        last_cp: Option<(u32, BlockHash)>,
        blocks: BTreeMap<u32, Block>,
    },
    Mempool(Vec<Transaction>),
    Synced,
}

#[derive(Debug)]
pub enum RpcError {
    Rpc(bitcoincore_rpc::Error),
    Send(SendError<RpcData>),
    Reorg(u32),
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
    tx_emit_threshold: usize,
}

impl Client {
    pub fn new(url: &str, auth: Auth) -> Result<Self, RpcError> {
        let client = RpcClient::new(url, auth)?;
        Ok(Client {
            client,
            tx_emit_threshold: TX_EMIT_THRESHOLD,
        })
    }

    pub fn emit_blocks(
        &self,
        chan: &SyncSender<RpcData>,
        local_cps: &mut BTreeMap<u32, BlockHash>,
        cp_limit: usize,
        fallback_height: u32,
    ) -> Result<(), RpcError> {
        let tip = self.client.get_block_count()? as u32;

        let local_tip = local_cps
            .iter()
            .next_back()
            .map(|(&height, _)| height as u32)
            .unwrap_or(fallback_height);

        chan.send(RpcData::Start {
            target_tip: tip as _,
            local_tip: local_tip as _,
        })?;

        // nothing to do if local tip is higher than node
        if local_tip > tip {
            return Ok(());
        }

        let mut last_agreement = None;
        let mut must_include = None;

        for (height, hash) in local_cps.iter().rev() {
            match self.client.get_block_info(hash) {
                Ok(res) => {
                    if res.confirmations < 0 {
                        must_include = Some(res.height as u32); // NOT in main chain
                    } else {
                        last_agreement = Some(res);
                        break;
                    }
                }
                Err(err) => {
                    use bitcoincore_rpc::jsonrpc;
                    match err {
                        bitcoincore_rpc::Error::JsonRpc(jsonrpc::Error::Rpc(rpc_err))
                            if rpc_err.code == -5 =>
                        {
                            must_include = Some(*height); // NOT in main chain
                        }
                        err => return Err(err.into()),
                    }
                }
            };
        }

        match &last_agreement {
            Some(res) => {
                println!("agreement @ height={}", res.height);
                local_cps.split_off(&((res.height + 1) as _));
            }
            None => {
                println!("no agreement, fallback_height={}", fallback_height);
            }
        };

        // batch of blocks to emit
        let mut to_emit = BTreeMap::<u32, Block>::new();

        // determine first block and last checkpoint that should be included (if any)
        let mut block = match last_agreement {
            Some(res) => match res.nextblockhash {
                Some(block_hash) => self.client.get_block_info(&block_hash)?,
                // no next block after agreement point, checkout mempool
                None => return self.emit_mempool(chan),
            },
            None => {
                let block_hash = self.client.get_block_hash(fallback_height as _)?;
                self.client.get_block_info(&block_hash)?
            }
        };

        let mut has_next = true;

        while has_next {
            if block.confirmations < 0 {
                return Err(RpcError::Reorg(block.height as _));
            }

            let _displaced = to_emit.insert(block.height as _, self.client.get_block(&block.hash)?);
            debug_assert_eq!(_displaced, None);

            match block.nextblockhash {
                Some(next_hash) => block = self.client.get_block_info(&next_hash)?,
                None => has_next = false,
            };

            if !has_next
                || must_include.as_ref() <= to_emit.keys().next_back()
                    && to_emit.iter().map(|(_, b)| b.txdata.len()).sum::<usize>()
                        >= self.tx_emit_threshold
            {
                let last_cp = local_cps.iter().next_back().map(|(h, b)| (*h, *b));
                let blocks = to_emit.split_off(&0);

                // update local checkpoints
                for (height, block) in &blocks {
                    local_cps.insert(*height, block.block_hash());
                }

                // prune local checkpoints
                if let Some(&last_height) = local_cps.keys().nth_back(cp_limit) {
                    let mut split = local_cps.split_off(&(last_height + 1));
                    core::mem::swap(local_cps, &mut split);
                }

                chan.send(RpcData::Blocks { last_cp, blocks })?;
            }
        }

        self.emit_mempool(chan)
    }

    pub fn emit_mempool(&self, chan: &SyncSender<RpcData>) -> Result<(), RpcError> {
        let ordered_txids = self
            .client
            .get_raw_mempool()?
            .into_iter()
            .map(|txid| {
                self.client
                    .get_mempool_entry(&txid)
                    .map(|entry| ((entry.depends.len(), txid), entry.depends))
            })
            .collect::<Result<BTreeMap<(usize, Txid), Vec<Txid>>, _>>()?;

        let mut done = HashSet::<Txid>::with_capacity(ordered_txids.len());

        while done.len() < ordered_txids.len() {
            let mut batch = Vec::new();

            for ((_, txid), depends) in &ordered_txids {
                if done.contains(txid) || depends.iter().any(|txid| !done.contains(txid)) {
                    continue;
                }

                let tx = self.client.get_raw_transaction(txid, None)?;
                batch.push(tx);
                done.insert(*txid);
            }

            chan.send(RpcData::Mempool(batch))?;
        }

        chan.send(RpcData::Synced)?;
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
