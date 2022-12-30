use std::{collections::BTreeMap, ops::Deref};

use bdk_chain::{
    bitcoin::{BlockHash, Script, Txid},
    sparse_chain::{InsertTxErr, SparseChain},
    BlockId, TxHeight,
};
use bdk_cli::Broadcast;
use electrum_client::{Client, Config, ElectrumApi};

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
    client: Client,
}

impl ElectrumClient {
    pub fn new(url: &str) -> Result<Self, ElectrumError> {
        let client = Client::from_config(url, Config::default())?;
        Ok(Self { client })
    }
}

impl Deref for ElectrumClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl Broadcast for ElectrumClient {
    type Error = ElectrumError;
    fn broadcast(&self, tx: &bdk_chain::bitcoin::Transaction) -> Result<(), Self::Error> {
        let _ = self.client.transaction_broadcast(tx)?;
        Ok(())
    }
}

impl ElectrumClient {
    /// Fetch latest block height.
    pub fn get_tip(&self) -> Result<(u32, BlockHash), ElectrumError> {
        // TODO: unsubscribe when added to the client, or is there a better call to use here?
        Ok(self
            .client
            .block_headers_subscribe()
            .map(|data| (data.height as u32, data.header.block_hash()))?)
    }

    /// Scan for a given list of scripts, and create an initial [`bdk_core::sparse_chain::SparseChain`] update candidate.
    /// This will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and create the final [`bdk_core::keychain::KeychainChangeSet`] before applying it.
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

    /// Scan for a keychain tracker, and create an initial [`bdk_core::sparse_chain::SparseChain`] update candidate.
    /// This will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and create the final [`bdk_core::keychain::KeychainChangeSet`] before applying it.
    pub fn wallet_txid_scan<K: Ord + Clone>(
        &self,
        scripts: BTreeMap<K, impl Iterator<Item = (u32, Script)>>,
        stop_gap: Option<usize>,
        local_chain: &BTreeMap<u32, BlockHash>,
        batch_size: usize,
    ) -> Result<(SparseChain, BTreeMap<K, u32>), ElectrumError> {
        let mut sparse_chain = SparseChain::default();

        // Check for reorgs.
        // In case of reorg, new checkpoints until the last common checkpoint is added to the structure
        for (&existing_height, &existing_hash) in local_chain.iter().rev() {
            let current_hash = self
                .client
                .block_header(existing_height as usize)?
                .block_hash();
            sparse_chain
                .insert_checkpoint(BlockId {
                    height: existing_height,
                    hash: current_hash,
                })
                .expect("This never errors because we are working with a fresh chain");

            if current_hash == existing_hash {
                break;
            }
        }

        // Insert the new tip
        let (tip_height, tip_hash) = self.get_tip()?;
        if sparse_chain
            .insert_checkpoint(BlockId {
                height: tip_height,
                hash: tip_hash,
            })
            .is_err()
        {
            // There has been an reorg since the last call to `block_header` in the loop above at the current tip.
            return Err(ElectrumError::Reorg);
        }

        let mut keychain_index_update = BTreeMap::new();

        // Fetch Keychain's last_active_index and all related txids.
        // Add them into the SparseChain
        for (keychain, mut scripts) in scripts {
            let mut last_active_index = 0;
            let mut unused_script_count = 0usize;

            loop {
                let batch_scripts = (0..batch_size)
                    .map(|_| scripts.next())
                    .filter_map(|item| item)
                    .collect::<Vec<_>>();

                if batch_scripts.is_empty() {
                    // We reached the end of the script list
                    break;
                }

                for (history, index) in self
                    .batch_script_get_history(batch_scripts.iter().map(|(_, script)| script))?
                    .iter()
                    .zip(batch_scripts.iter().map(|(index, _)| index))
                {
                    let txid_list = history
                        .iter()
                        .map(|history_result| {
                            if history_result.height > 0
                                && (history_result.height as u32) <= tip_height
                            {
                                return (
                                    history_result.tx_hash,
                                    TxHeight::Confirmed(history_result.height as u32),
                                );
                            } else {
                                return (history_result.tx_hash, TxHeight::Unconfirmed);
                            };
                        })
                        .collect::<Vec<(Txid, TxHeight)>>();

                    if txid_list.is_empty() {
                        unused_script_count += 1;
                    } else {
                        last_active_index = *index;
                        unused_script_count = 0;
                    }

                    for (txid, index) in txid_list {
                        if let Err(err) = sparse_chain.insert_tx(txid, index) {
                            match err {
                                InsertTxErr::TxTooHigh => {
                                    unreachable!("We should not encounter this error as we ensured TxHeight <= tip_height");
                                }
                                InsertTxErr::TxMoved => {
                                    /* This means there is a reorg, we will handle this situation below */
                                }
                            }
                        }
                    }
                }
            }

            if unused_script_count >= stop_gap.unwrap_or(usize::MAX) {
                break;
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
