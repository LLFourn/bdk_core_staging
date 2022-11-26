use bdk_core::{
    bitcoin::{BlockHash, Script, Transaction},
    chain_graph::ChainGraph,
    keychain::KeychainScan,
    sparse_chain::{InsertCheckpointErr, InsertTxErr},
    BlockId, ConfirmationTime,
};
use std::collections::BTreeMap;

use esplora_client::{BlockingClient, Builder};

use bdk_cli::Result;

#[derive(Debug, Clone)]
pub struct Client {
    pub parallel_requests: u8,
    pub client: BlockingClient,
}

impl Client {
    /// TODO
    pub fn new(base_url: &str, parallel_requests: u8) -> Result<Self> {
        Ok(Self {
            parallel_requests,
            client: Builder::new(base_url).build_blocking()?,
        })
    }

    pub fn spk_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        existing_chain: BTreeMap<u32, BlockHash>,
    ) -> Result<ChainGraph<ConfirmationTime>> {
        let mut dummy_keychains = BTreeMap::new();
        dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

        let wallet_scan = self.wallet_scan(dummy_keychains, None, existing_chain)?;

        Ok(wallet_scan.update)
    }

    /// Create a new checkpoint with transactions spending from or to the scriptpubkeys in
    /// `scripts`.
    pub fn wallet_scan<K: Ord + Clone, I>(
        &self,
        keychains: BTreeMap<K, I>,
        stop_gap: Option<usize>,
        existing_chain: BTreeMap<u32, BlockHash>,
    ) -> Result<KeychainScan<K, ConfirmationTime>>
    where
        I: Iterator<Item = (u32, Script)>,
    {
        let mut wallet_scan = KeychainScan::default();
        let update = &mut wallet_scan.update;

        for (&existing_height, &existing_hash) in existing_chain.iter().rev() {
            let current_hash = self.client.get_block_hash(existing_height)?;
            update
                .insert_checkpoint(BlockId {
                    height: existing_height,
                    hash: current_hash,
                })
                .expect("should not collide");

            if current_hash == existing_hash {
                break;
            }
        }

        let tip_at_start = BlockId {
            height: self.client.get_height()?,
            hash: self.client.get_tip_hash()?,
        };
        if let Err(err) = update.insert_checkpoint(tip_at_start) {
            match err {
                InsertCheckpointErr::HashNotMatching => {
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
                        if let Err(err) = update.insert_tx(tx.to_tx(), confirmation_time) {
                            match err {
                                InsertTxErr::TxTooHigh => {
                                    /* Don't care about new transactions confirmed while syncing */
                                }
                                InsertTxErr::TxMoved => {
                                    /* This means there is a reorg, we will catch that below */
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
        for end_block in self
            .client
            .get_recent_blocks(None)?
            .iter()
            .map(|esplora_block| BlockId {
                height: esplora_block.height,
                hash: esplora_block.id,
            })
        {
            update.insert_checkpoint(end_block)?;
        }

        Ok(wallet_scan)
    }
}

impl bdk_cli::Broadcast for Client {
    fn broadcast(&self, tx: &Transaction) -> bdk_cli::Result<()> {
        Ok(self.client.broadcast(tx)?)
    }
}
