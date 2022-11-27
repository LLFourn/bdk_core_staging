use std::{collections::BTreeMap, ops::Deref};

use bdk_cli::{Broadcast, Result};
use bdk_core::{
    bitcoin::{Script, Txid},
    chain_graph::ChainGraph,
    BlockId, TxHeight,
};
use bdk_keychain::KeychainScan;
use electrum_client::{Client, Config, ElectrumApi};

pub struct ElectrumClient {
    client: Client,
}

impl ElectrumClient {
    pub fn new(url: &str) -> Result<Self> {
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
    fn broadcast(&self, tx: &bdk_core::bitcoin::Transaction) -> Result<()> {
        let _ = self.client.transaction_broadcast(tx)?;
        Ok(())
    }
}

/// We will detect reorg if it has depth less than this
const REORG_DETECTION_DEPTH: u32 = 100;
const DEFAULT_STOP_GAP: u32 = 10;

impl ElectrumClient {
    /// Fetch latest block height.
    pub fn get_height(&self) -> Result<u32> {
        // TODO: unsubscribe when added to the client, or is there a better call to use here?
        Ok(self
            .client
            .block_headers_subscribe()
            .map(|data| data.height as u32)?)
    }

    /// Scan for a list of scripts, and create an initial [`ChainGraph`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`ChainGraph`] before applying it.
    pub fn spk_txid_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        last_known_height: Option<u32>,
    ) -> Result<ChainGraph<TxHeight>> {
        let mut dummy_keychains = BTreeMap::new();
        dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

        let wallet_scan = self.wallet_txid_scan(dummy_keychains, None, last_known_height)?;

        Ok(wallet_scan.update)
    }

    /// Scan for a keychain tracker, and create an initial [`KeychainScan`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`KeychainScan`] before applying it.
    pub fn wallet_txid_scan<K: Ord + Clone>(
        &self,
        keychains: BTreeMap<K, impl Iterator<Item = (u32, Script)>>,
        stop_gap: Option<u32>,
        last_known_height: Option<u32>,
    ) -> Result<KeychainScan<K, TxHeight>> {
        let mut keychain_scan = KeychainScan::default();

        // 1. Create checkpoint data from last_known_height - REORG_DEPTH
        // Reorg situation will he handled at the time of applying this KeychainScan.
        // If there's reorg deeper than the assumed depth, update process will throw error.
        let current_height = self.get_height()?;
        let check_from = last_known_height
            .map(|ht| ht.saturating_sub(REORG_DETECTION_DEPTH))
            .unwrap_or(0);
        let required_block_count = (current_height - check_from) + 1;
        let headers = self.block_headers(check_from as usize, required_block_count as usize)?;

        let block_ids =
            (check_from..=current_height)
                .zip(headers.headers)
                .map(|(height, header)| BlockId {
                    height,
                    hash: header.block_hash(),
                });

        for block in block_ids {
            keychain_scan.update.chain.insert_checkpoint(block)?;
        }

        // 2. Fetch Keychain's last_active_index and all related txids.
        // Add them into the KeyChainScan
        for (keychain, mut scripts) in keychains.into_iter() {
            let mut last_active_index = 0;
            let mut unused_script_count = 0u32;
            let mut script_history_txid = Vec::<(Txid, TxHeight)>::new();

            loop {
                let (index, script) = scripts.next().expect("its an infinite iterator");

                let history = self
                    .script_get_history(&script)?
                    .iter()
                    .map(|history_result| {
                        if history_result.height > 0 {
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
                }

                if unused_script_count >= stop_gap.unwrap_or(DEFAULT_STOP_GAP) {
                    break;
                }
            }

            for (txid, index) in script_history_txid {
                keychain_scan.update.chain.insert_tx(txid, index)?;
            }
            // TODO: Add log
            keychain_scan
                .last_active_indexes
                .insert(keychain, last_active_index);
        }

        Ok(keychain_scan)
    }
}
