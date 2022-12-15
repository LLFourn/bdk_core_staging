use std::{collections::BTreeMap, ops::Deref};

use bdk_cli::Broadcast;
use bdk_core::{
    bitcoin::{BlockHash, Script, Txid},
    sparse_chain::{InsertCheckpointErr, InsertTxErr, SparseChain, UpdateFailure},
    BlockId, TxHeight,
};
use electrum_client::{Client, Config, ElectrumApi};

#[derive(Debug)]
pub enum ElectrumError {
    Client(electrum_client::Error),
    InsertTx(InsertTxErr),
    Update(UpdateFailure),
    InsertCheckpoint(InsertCheckpointErr),
}

impl core::fmt::Display for ElectrumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElectrumError::Client(e) => write!(f, "{}", e),
            ElectrumError::InsertTx(e) => write!(f, "{}", e),
            ElectrumError::Update(e) => write!(f, "{}", e),
            ElectrumError::InsertCheckpoint(e) => write!(f, "{}", e),
        }
    }
}

impl std::error::Error for ElectrumError {}

impl From<electrum_client::Error> for ElectrumError {
    fn from(e: electrum_client::Error) -> Self {
        Self::Client(e)
    }
}

impl From<InsertTxErr> for ElectrumError {
    fn from(e: InsertTxErr) -> Self {
        Self::InsertTx(e)
    }
}

impl From<UpdateFailure> for ElectrumError {
    fn from(e: UpdateFailure) -> Self {
        Self::Update(e)
    }
}

impl From<InsertCheckpointErr> for ElectrumError {
    fn from(e: InsertCheckpointErr) -> Self {
        Self::InsertCheckpoint(e)
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
    fn broadcast(&self, tx: &bdk_core::bitcoin::Transaction) -> Result<(), Self::Error> {
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

    /// Scan for a list of scripts, and create an initial [`ChainGraph`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`ChainGraph`] before applying it.
    pub fn spk_txid_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        local_chain: &BTreeMap<u32, BlockHash>,
    ) -> Result<SparseChain, ElectrumError> {
        let mut dummy_keychains = BTreeMap::new();
        dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

        Ok(self.wallet_txid_scan(dummy_keychains, None, local_chain)?.0)
    }

    /// Scan for a keychain tracker, and create an initial [`KeychainScan`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`KeychainScan`] before applying it.
    pub fn wallet_txid_scan<K: Ord + Clone>(
        &self,
        scripts: BTreeMap<K, impl Iterator<Item = (u32, Script)>>,
        stop_gap: Option<usize>,
        local_chain: &BTreeMap<u32, BlockHash>,
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
        sparse_chain
            .insert_checkpoint(BlockId {
                height: tip_height,
                hash: tip_hash,
            })
            .expect("This never errors because we are working with a fresh chain");

        let mut keychain_index_update = BTreeMap::new();

        let mut reorgred_tx = Vec::new();

        // Fetch Keychain's last_active_index and all related txids.
        // Add them into the KeyChainScan
        for (keychain, mut scripts) in scripts.into_iter() {
            let mut last_active_index = 0;
            let mut unused_script_count = 0usize;
            let mut script_history_txid = Vec::<(Txid, TxHeight)>::new();

            loop {
                if let Some((index, script)) = scripts.next() {
                    let history = self
                        .script_get_history(&script)?
                        .iter()
                        .map(|history_result| {
                            if history_result.height > 0
                                && (history_result.height as u32) < tip_height
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

                    if history.is_empty() {
                        unused_script_count += 1;
                    } else {
                        last_active_index = index;
                        script_history_txid.extend(history.iter());
                        unused_script_count = 0;
                    }

                    if unused_script_count >= stop_gap.unwrap_or(usize::MAX) {
                        break;
                    }
                } else {
                    break;
                }
            }

            for (txid, index) in script_history_txid {
                if let Err(err) = sparse_chain.insert_tx(txid, index) {
                    match err {
                        InsertTxErr::TxTooHigh => {
                            /* Don't care about new transactions confirmed while syncing */
                        }
                        InsertTxErr::TxMoved => {
                            /* This means there is a reorg, we will handle this situation below */
                            reorgred_tx.push((txid, index));
                        }
                    }
                }
            }

            keychain_index_update.insert(keychain, last_active_index);
        }

        // To handle reorgs during syncing we re-apply last known
        // 20 blocks again into sparsechain
        let mut reorged_sparse_chain = SparseChain::default();
        let (tip_height, _) = self.get_tip()?;
        let reorged_headers = self
            .block_headers((tip_height - 20) as usize, 21)?
            .headers
            .into_iter()
            .map(|header| header.block_hash())
            .zip((tip_height - 20)..=tip_height);

        // Insert the new checkpoints
        for (hash, height) in reorged_headers {
            reorged_sparse_chain.insert_checkpoint(BlockId { height, hash })?;
        }

        // Insert reorged txs
        for (txid, index) in reorgred_tx {
            reorged_sparse_chain.insert_tx(txid, index)?;
        }

        // Sync the original sparse_chain with new reorged_sparse_chain
        sparse_chain.apply_update(reorged_sparse_chain)?;

        Ok((sparse_chain, keychain_index_update))
    }
}
