use std::{collections::BTreeMap, ops::Deref};

use bdk_chain::{
    bitcoin::{BlockHash, Script, Txid},
    sparse_chain::{self, SparseChain},
    BlockId, TxHeight,
};
use bdk_cli::Broadcast;
use electrum_client::{Client, ElectrumApi};

#[derive(Debug)]
pub enum ElectrumError {
    Client(electrum_client::Error),
    Reorg,
}

impl core::fmt::Display for ElectrumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElectrumError::Client(e) => write!(f, "{}", e),
            ElectrumError::Reorg => write!(
                f,
                "Reorg detected at sync time. Please run the sync call again"
            ),
        }
    }
}

impl std::error::Error for ElectrumError {}

impl From<electrum_client::Error> for ElectrumError {
    fn from(e: electrum_client::Error) -> Self {
        Self::Client(e)
    }
}

pub struct ElectrumClient {
    inner: Client,
}

impl ElectrumClient {
    pub fn new(client: Client) -> Result<Self, ElectrumError> {
        Ok(Self { inner: client })
    }
}

impl Deref for ElectrumClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Broadcast for ElectrumClient {
    type Error = electrum_client::Error;
    fn broadcast(&self, tx: &bdk_chain::bitcoin::Transaction) -> Result<(), Self::Error> {
        let _ = self.inner.transaction_broadcast(tx)?;
        Ok(())
    }
}

impl ElectrumClient {
    /// Fetch latest block height.
    pub fn get_tip(&self) -> Result<(u32, BlockHash), electrum_client::Error> {
        // TODO: unsubscribe when added to the client, or is there a better call to use here?
        Ok(self
            .inner
            .block_headers_subscribe()
            .map(|data| (data.height as u32, data.header.block_hash()))?)
    }

    /// Scan for a given list of scripts, and create an initial [`bdk_chain::sparse_chain::SparseChain`] update candidate.
    /// This will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and create the final [`bdk_chain::keychain::KeychainChangeSet`] before applying it.
    pub fn spk_txid_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        local_chain: &BTreeMap<u32, BlockHash>,
        batch_size: usize,
    ) -> Result<SparseChain, ElectrumError> {
        let mut dummy_keychains = BTreeMap::new();
        dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

        Ok(self
            .wallet_txid_scan(dummy_keychains, None, local_chain, batch_size)?
            .0)
    }

    /// Scan for a keychain tracker, and create an initial [`bdk_chain::sparse_chain::SparseChain`] update candidate.
    /// This will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and create the final [`bdk_chain::keychain::KeychainChangeSet`] before applying it.
    pub fn wallet_txid_scan<K: Ord + Clone>(
        &self,
        scripts: BTreeMap<K, impl Iterator<Item = (u32, Script)>>,
        stop_gap: Option<usize>,
        local_chain: &BTreeMap<u32, BlockHash>,
        batch_size: usize,
    ) -> Result<(SparseChain, BTreeMap<K, u32>), ElectrumError> {
        let mut sparse_chain = SparseChain::default();

        // Find local chain block that is still there so our update can connect to the local chain.
        for (&existing_height, &existing_hash) in local_chain.iter().rev() {
            let current_hash = self
                .inner
                .block_header(existing_height as usize)?
                .block_hash();
            let changeset = sparse_chain
                .insert_checkpoint_preview(BlockId {
                    height: existing_height,
                    hash: current_hash,
                })
                .expect("This never errors because we are working with a fresh chain");
            sparse_chain.apply_changeset(changeset);

            if current_hash == existing_hash {
                break;
            }
        }

        // Insert the new tip so new transactions will be accepted into the sparse chain.
        let tip = {
            let (height, hash) = self.get_tip()?;
            BlockId { height, hash }
        };
        if let Err(failure) = sparse_chain.insert_checkpoint(tip) {
            match failure {
                sparse_chain::InsertCheckpointError::HashNotMatching { .. } => {
                    // There has been a re-org before we even begin scanning addresses.
                    // Just recursively call (this should never happen).
                    return self.wallet_txid_scan(scripts, stop_gap, local_chain, batch_size);
                }
            }
        }

        let mut keychain_index_update = BTreeMap::new();

        for (keychain, mut scripts) in scripts {
            let mut last_active_index = 0;
            let mut unused_script_count = 0usize;

            loop {
                let mut next_batch = (0..batch_size).filter_map(|_| scripts.next()).peekable();

                if next_batch.peek().is_none() {
                    break;
                }

                let (indexes, scripts): (Vec<_>, Vec<_>) = next_batch.unzip();

                for (history, index) in self
                    .batch_script_get_history(scripts.iter())?
                    .into_iter()
                    .zip(indexes)
                {
                    let txid_list = history
                        .iter()
                        .map(|history_result| {
                            if history_result.height > 0
                                && (history_result.height as u32) <= tip.height
                            {
                                (
                                    history_result.tx_hash,
                                    TxHeight::Confirmed(history_result.height as u32),
                                )
                            } else {
                                (history_result.tx_hash, TxHeight::Unconfirmed)
                            }
                        })
                        .collect::<Vec<(Txid, TxHeight)>>();

                    if txid_list.is_empty() {
                        unused_script_count += 1;
                    } else {
                        if index > last_active_index {
                            last_active_index = index;
                        }
                        unused_script_count = 0;
                    }

                    for (txid, pos) in txid_list {
                        if let Err(failure) = sparse_chain.insert_tx(txid, pos) {
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
                }

                if unused_script_count >= stop_gap.unwrap_or(usize::MAX) {
                    break;
                }
            }

            keychain_index_update.insert(keychain, last_active_index);
        }

        // Check for Reorg during the above sync process
        let our_latest = sparse_chain.latest_checkpoint().expect("must exist");
        if our_latest.hash != self.block_header(our_latest.height as usize)?.block_hash() {
            return Err(ElectrumError::Reorg);
        }

        Ok((sparse_chain, keychain_index_update))
    }
}
