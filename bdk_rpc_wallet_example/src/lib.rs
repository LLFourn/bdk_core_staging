use bdk_chain::{
    bitcoin::{BlockHash, Transaction, Txid},
    sparse_chain::{self, SparseChain},
    BlockId, TxHeight,
};
use bdk_cli::Broadcast;
use bitcoincore_rpc::{bitcoincore_rpc_json::ImportMultiResultError, Client, RpcApi};

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Debug,
    ops::Deref,
};

/// Bitcoin Core RPC related errors.
#[derive(Debug)]
pub enum RpcError {
    Client(bitcoincore_rpc::Error),
    ImportMulti(ImportMultiResultError),
    General(String),
}

impl core::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RpcError::Client(e) => write!(f, "{}", e),
            RpcError::ImportMulti(e) => write!(f, "{:?}", e),
            RpcError::General(s) => write!(f, "{}", s),
        }
    }
}

impl std::error::Error for RpcError {}

impl From<bitcoincore_rpc::Error> for RpcError {
    fn from(e: bitcoincore_rpc::Error) -> Self {
        Self::Client(e)
    }
}

impl From<ImportMultiResultError> for RpcError {
    fn from(value: ImportMultiResultError) -> Self {
        Self::ImportMulti(value)
    }
}

pub struct RpcClient {
    pub client: Client,
}

impl Deref for RpcClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl RpcClient {
    pub fn get_tip(&self) -> Result<(u64, BlockHash), RpcError> {
        Ok(self
            .client
            .get_blockchain_info()
            .map(|i| (i.blocks, i.best_block_hash))?)
    }

    /// Scan for a keychain tracker, and create an initial [`SparseChain`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`SparseChain`] before applying it.
    pub fn wallet_scan(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
    ) -> Result<SparseChain, RpcError> {
        let mut sparse_chain = SparseChain::default();

        let mut last_common_height = 0;
        for (&height, &original_hash) in local_chain.iter().rev() {
            let update_block_id = BlockId {
                height,
                hash: self.client.get_block_hash(height as u64)?,
            };
            let _ = sparse_chain
                .insert_checkpoint(update_block_id)
                .expect("should not collide");
            if update_block_id.hash == original_hash {
                last_common_height = update_block_id.height;
                break;
            }
        }

        // Insert the new tip so new transactions will be accepted into the sparse chain.
        let tip = self.get_tip().map(|(height, hash)| BlockId {
            height: height as u32,
            hash,
        })?;
        if let Err(failure) = sparse_chain.insert_checkpoint(tip) {
            match failure {
                sparse_chain::InsertCheckpointError::HashNotMatching { .. } => {
                    // There has been a re-org before we even begin scanning addresses.
                    // Just recursively call (this should never happen).
                    return self.wallet_scan(local_chain);
                }
            }
        }

        // Fetch the transactions
        let page_size = 1000; // Core has 1000 page size limit
        let mut page = 0;

        let _ = self
            .client
            .rescan_blockchain(Some(last_common_height as usize), None);

        let mut txids_to_update = Vec::new();

        loop {
            let list_tx_result = self.client
                .list_transactions(None, Some(page_size), Some(page * page_size), Some(true))?
                .iter()
                .filter(|item|
                    // filter out conflicting transactions - only accept transactions that are already
                    // confirmed, or exists in mempool
                    item.info.confirmations > 0 || self.client.get_mempool_entry(&item.info.txid).is_ok())
                .map(|list_result| {
                    let chain_index = match list_result.info.blockheight {
                        Some(height) if height <= tip.height => TxHeight::Confirmed(height),
                        _ => TxHeight::Unconfirmed,
                    };
                    (chain_index, list_result.info.txid)
                })
                .collect::<HashSet<(TxHeight, Txid)>>();

            txids_to_update.extend(list_tx_result.iter());

            if list_tx_result.len() < page_size {
                break;
            }
            page += 1;
        }

        for (index, txid) in txids_to_update {
            if let Err(failure) = sparse_chain.insert_tx(txid, index) {
                match failure {
                    sparse_chain::InsertTxError::TxTooHigh { .. } => {
                        unreachable!("We should not encounter this error as we ensured tx_height <= tip.height");
                    }
                    sparse_chain::InsertTxError::TxMovedUnexpectedly { .. } => {
                        /* This means there is a reorg, we will handle this situation below */
                    }
                }
            }
        }

        // Check for Reorg during the above sync process, recursively call the scan if detected.
        let our_latest = sparse_chain.latest_checkpoint().expect("must exist");
        if our_latest.hash != self.client.get_block_hash(our_latest.height as u64)? {
            return self.wallet_scan(local_chain);
        }

        Ok(sparse_chain)
    }
}

impl Broadcast for RpcClient {
    type Error = bitcoincore_rpc::Error;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error> {
        let _ = self.client.send_raw_transaction(tx)?;
        Ok(())
    }
}
