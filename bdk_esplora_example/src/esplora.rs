use bdk_chain::{
    bitcoin::{BlockHash, Script, Transaction},
    chain_graph::ChainGraph,
    keychain::KeychainScan,
    sparse_chain, BlockId, ConfirmationTime,
};
use esplora_client::{BlockingClient, Builder};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Client {
    pub parallel_requests: u8,
    pub client: BlockingClient,
}

#[derive(Debug)]
pub enum UpdateError {
    Client(esplora_client::Error),
    Reorg,
}

impl core::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::Client(e) => write!(f, "{}", e),
            UpdateError::Reorg => write!(f, "Reorg occurred while the sync was in progress",),
        }
    }
}

impl From<esplora_client::Error> for UpdateError {
    fn from(value: esplora_client::Error) -> Self {
        UpdateError::Client(value)
    }
}

impl std::error::Error for UpdateError {}

impl Client {
    /// Creates a new client that makes requests to `base_url`
    pub fn new(base_url: &str, parallel_requests: u8) -> Result<Self, esplora_client::Error> {
        Ok(Self {
            parallel_requests,
            client: Builder::new(base_url).build_blocking()?,
        })
    }

    /// Scans an iterator of script pubkeys for transactions spending to or from them.
    ///
    /// Stops after a gap of `stop_gap` script pubkeys with no associated transactions.
    pub fn spk_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        local_chain: &BTreeMap<u32, BlockHash>,
        stop_gap: Option<usize>,
    ) -> Result<ChainGraph<ConfirmationTime>, UpdateError> {
        let mut dummy_keychains = BTreeMap::new();
        dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

        let wallet_scan = self.wallet_scan(dummy_keychains, local_chain, stop_gap)?;

        Ok(wallet_scan.update)
    }

    /// Scans several iterators of script pubkeys for transactions spending to or from them.
    ///
    /// The scan for each keychain stops after a gap of `stop_gap` script pubkeys with no associated
    /// transactions.
    pub fn wallet_scan<K: Ord + Clone, I>(
        &self,
        keychains: BTreeMap<K, I>,
        local_chain: &BTreeMap<u32, BlockHash>,
        stop_gap: Option<usize>,
    ) -> Result<KeychainScan<K, ConfirmationTime>, UpdateError>
    where
        I: Iterator<Item = (u32, Script)>,
    {
        let mut wallet_scan = KeychainScan::default();
        let update = &mut wallet_scan.update;

        for (&height, &original_hash) in local_chain.iter().rev() {
            let update_block_id = BlockId {
                height,
                hash: self.client.get_block_hash(height)?,
            };
            let _ = update
                .insert_checkpoint(update_block_id)
                .expect("should not collide");
            if update_block_id.hash == original_hash {
                break;
            }
        }

        let tip_at_start = BlockId {
            height: self.client.get_height()?,
            hash: self.client.get_tip_hash()?,
        };
        if let Err(failure) = update.insert_checkpoint(tip_at_start) {
            match failure {
                sparse_chain::InsertCheckpointError::HashNotMatching { .. } => {
                    /* There has been a reorg since the line of code above, we will catch this later on */
                }
            }
        }

        for (keychain, mut spks) in keychains {
            let mut last_active_index = None;
            let mut empty_scripts = 0;

            loop {
                let handles = (0..self.parallel_requests)
                    .filter_map(
                        |_| -> Option<
                            std::thread::JoinHandle<Result<(u32, Vec<esplora_client::Tx>), _>>,
                        > {
                            let (index, script) = spks.next()?;
                            let client = self.client.clone();
                            Some(std::thread::spawn(move || {
                                let mut related_txs = client.scripthash_txs(&script, None)?;

                                let n_confirmed =
                                    related_txs.iter().filter(|tx| tx.status.confirmed).count();
                                // esplora pages on 25 confirmed transactions. If there's 25 or more we
                                // keep requesting to see if there's more.
                                if n_confirmed >= 25 {
                                    loop {
                                        let new_related_txs = client.scripthash_txs(
                                            &script,
                                            Some(related_txs.last().unwrap().txid),
                                        )?;
                                        let n = new_related_txs.len();
                                        related_txs.extend(new_related_txs);
                                        // we've reached the end
                                        if n < 25 {
                                            break;
                                        }
                                    }
                                }

                                Result::<_, esplora_client::Error>::Ok((index, related_txs))
                            }))
                        },
                    )
                    .collect::<Vec<_>>();

                let n_handles = handles.len();

                for handle in handles {
                    let (index, related_txs) = handle.join().unwrap()?; // TODO: don't unwrap
                    if related_txs.is_empty() {
                        empty_scripts += 1;
                    } else {
                        last_active_index = Some(index);
                        empty_scripts = 0;
                    }
                    for tx in related_txs {
                        let confirmation_time = match tx.status.confirmed {
                            true => ConfirmationTime::Confirmed {
                                height: tx.status.block_height.expect("height expected"),
                                time: tx.status.block_time.expect("blocktime expected"),
                            },
                            false => ConfirmationTime::Unconfirmed,
                        };
                        if let Err(failure) = update.insert_tx(tx.to_tx(), confirmation_time) {
                            use bdk_chain::{
                                chain_graph::InsertTxError, sparse_chain::InsertTxError::*,
                            };
                            match failure {
                                InsertTxError::Chain(TxTooHigh { .. }) => {
                                    /* Chain tip has increased, ignore tx for now */
                                }
                                InsertTxError::Chain(TxMovedUnexpectedly { .. }) => {
                                    /* Reorg occured (catch error below), ignore tx for now */
                                }
                                InsertTxError::UnresolvableConflict(_) => {
                                    /* Reorg occured (catch error below), ignore tx for now */
                                }
                            }
                        }
                    }
                }

                if n_handles == 0 || empty_scripts >= stop_gap.unwrap_or(usize::MAX) {
                    break;
                }
            }

            if let Some(last_active_index) = last_active_index {
                wallet_scan
                    .last_active_indexes
                    .insert(keychain, last_active_index);
            }
        }

        // Depending upon service providers number of recent blocks returned will vary.
        // esplora returns 10.
        // mempool.space returns 15.
        for block in self.client.get_recent_blocks(None)? {
            let block_id = BlockId {
                height: block.height,
                hash: block.id,
            };
            let _ = update
                .insert_checkpoint(block_id)
                .map_err(|_| UpdateError::Reorg)?;
        }

        Ok(wallet_scan)
    }
}

impl bdk_cli::Broadcast for Client {
    type Error = esplora_client::Error;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error> {
        Ok(self.client.broadcast(tx)?)
    }
}
