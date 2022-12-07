use std::{collections::BTreeMap, ops::Deref};

use bdk_cli::{anyhow::Result, Broadcast};
use bdk_core::{
    bitcoin::{BlockHash, Script, Txid},
    sparse_chain::SparseChain,
    BlockId, TxHeight,
};
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
    type Error = electrum_client::Error;
    fn broadcast(&self, tx: &bdk_core::bitcoin::Transaction) -> Result<(), Self::Error> {
        let _ = self.client.transaction_broadcast(tx)?;
        Ok(())
    }
}

impl ElectrumClient {
    /// Fetch latest block height.
    pub fn get_tip(&self) -> Result<(u32, BlockHash)> {
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
    ) -> Result<SparseChain> {
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
    ) -> Result<(SparseChain, BTreeMap<K, u32>)> {
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
                .expect("should not collide");

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
            .expect("Should not collide");

        let mut keychain_index_update = BTreeMap::new();

        // Fetch Keychain's last_active_index and all related txids.
        // Add them into the KeyChainScan
        for (keychain, mut scripts) in scripts.into_iter() {
            let mut last_active_index = 0;
            let mut unused_script_count = 0usize;
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
                    unused_script_count = 0;
                }

                if unused_script_count >= stop_gap.unwrap_or(usize::MAX) {
                    break;
                }
            }

            for (txid, index) in script_history_txid {
                sparse_chain.insert_tx(txid, index)?;
            }

            keychain_index_update.insert(keychain, last_active_index);
        }

        Ok((sparse_chain, keychain_index_update))
    }
}
