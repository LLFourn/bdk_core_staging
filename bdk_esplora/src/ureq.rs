use crate::api;
use bdk_core::{
    bitcoin::{
        consensus,
        hashes::{hex::ToHex, sha256, Hash},
        BlockHash, Script, Transaction, Txid,
    },
    chain_graph::ChainGraph,
    sparse_chain::{InsertCheckpointErr, InsertTxErr},
    BlockId, ConfirmationTime, KeychainScan,
};
use std::collections::{BTreeMap, BTreeSet};
pub use ureq;
use ureq::Agent;

#[derive(Debug, Clone)]
pub struct Client {
    pub parallel_requests: u8,
    pub base_url: String,
    pub agent: Agent,
}

#[derive(Debug)]
pub enum UpdateError {
    Ureq(ureq::Error),
    Deserialization { url: String },
    Reorg,
}

#[derive(Debug)]
pub enum Error {
    Ureq(ureq::Error),
    Deserialization { url: String },
}

impl From<Error> for UpdateError {
    fn from(e: Error) -> Self {
        match e {
            Error::Ureq(e) => UpdateError::Ureq(e),
            Error::Deserialization { url } => UpdateError::Deserialization { url },
        }
    }
}

impl core::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpdateError::Ureq(e) => write!(f, "{}", e),
            UpdateError::Deserialization { url } => {
                write!(f, "Failed to deserialize response from {}", url)
            }
            UpdateError::Reorg => write!(f, "Reorg occured while the sync was in progress",),
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Ureq(e) => write!(f, "{}", e),
            Error::Deserialization { url } => {
                write!(f, "Failed to deserialize response from {}", url)
            }
        }
    }
}

impl std::error::Error for Error {}
impl std::error::Error for UpdateError {}

impl From<ureq::Error> for UpdateError {
    fn from(e: ureq::Error) -> Self {
        UpdateError::Ureq(e)
    }
}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        Error::Ureq(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Ureq(e.into())
    }
}

impl Client {
    /// TODO
    pub fn new(agent: Agent, base_url: &str) -> Self {
        Self {
            agent,
            base_url: base_url.to_string(),
            parallel_requests: crate::DEFAULT_PARALLEL_REQUESTS,
        }
    }

    fn _scripthash_txs(
        &self,
        script: &Script,
        last_seen: Option<Txid>,
    ) -> Result<Vec<api::Tx>, ureq::Error> {
        let script_hash = sha256::Hash::hash(script.as_bytes()).into_inner().to_hex();
        let url = match last_seen {
            Some(last_seen) => format!(
                "{}/scripthash/{}/txs/chain/{}",
                self.base_url, script_hash, last_seen
            ),
            None => format!("{}/scripthash/{}/txs", self.base_url, script_hash),
        };
        Ok(self.agent.get(&url).call()?.into_json()?)
    }

    pub fn tip_hash(&self) -> Result<BlockHash, Error> {
        let url = format!("{}/blocks/tip/hash", self.base_url);
        let response = self.agent.get(&url).call()?;
        Ok(response
            .into_string()?
            .parse()
            .map_err(|_| Error::Deserialization { url })?)
    }

    pub fn tip(&self) -> Result<BlockId, Error> {
        let height = {
            let url = format!("{}/blocks/tip/height", self.base_url);
            let response = self.agent.get(&url).call()?;
            response
                .into_string()?
                .parse()
                .map_err(|_| Error::Deserialization { url })?
        };

        let hash = {
            let url = format!("{}/block-height/{}", self.base_url, height);
            let response = self.agent.get(&url).call()?;
            response
                .into_string()?
                .parse()
                .map_err(|_| Error::Deserialization { url })?
        };

        Ok(BlockId { height, hash })
    }

    fn recent_blocks(&self) -> Result<BTreeSet<BlockId>, ureq::Error> {
        let url = format!("{}/blocks", self.base_url);
        let response = self.agent.get(&url).call()?;
        let blocks: Vec<api::Block> = response.into_json()?;
        Ok(blocks
            .into_iter()
            .map(|block| BlockId {
                hash: block.id,
                height: block.height,
            })
            .collect())
    }

    fn block_hash_at_height(&self, height: u32) -> Result<BlockHash, Error> {
        let url = format!("{}/block-height/{}", self.base_url, height);
        let response = self.agent.get(&url).call()?;
        response
            .into_string()?
            .parse()
            .map_err(|_| Error::Deserialization { url })
    }

    pub fn broadcast(&self, tx: &Transaction) -> Result<(), ureq::Error> {
        let url = format!("{}/tx", self.base_url);
        let resp = self
            .agent
            .post(&url)
            .send_string(&consensus::serialize(tx).to_hex());
        // if let Err(e) = resp {
        //     dbg!(e.into_response().unwrap().into_string().unwrap());
        // }
        // TODO: make broadcast errors really good!
        resp?;
        Ok(())
    }

    pub fn spk_scan(
        &self,
        spks: impl Iterator<Item = Script>,
        existing_chain: BTreeMap<u32, BlockHash>,
    ) -> Result<ChainGraph<ConfirmationTime>, UpdateError> {
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
    ) -> Result<KeychainScan<K, ConfirmationTime>, UpdateError>
    where
        I: Iterator<Item = (u32, Script)>,
    {
        let mut wallet_scan = KeychainScan::default();
        let update = &mut wallet_scan.update;

        for (&existing_height, &existing_hash) in existing_chain.iter().rev() {
            let current_hash = self.block_hash_at_height(existing_height)?;
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

        let tip_at_start = self.tip()?;
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
                    .filter_map(|_| {
                        let (index, script) = spks.next()?;
                        let client = self.clone();
                        Some(std::thread::spawn(move || {
                            let mut related_txs: Vec<api::Tx> =
                                client._scripthash_txs(&script, None)?;

                            let n_confirmed =
                                related_txs.iter().filter(|tx| tx.status.confirmed).count();
                            // esplora pages on 25 confirmed transactions. If there's 25 or more we
                            // keep requesting to see if there's more.
                            if n_confirmed >= 25 {
                                loop {
                                    let new_related_txs: Vec<api::Tx> = client._scripthash_txs(
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

                            Result::<_, ureq::Error>::Ok((index, related_txs))
                        }))
                    })
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
                        if let Err(err) =
                            update.insert_tx(tx.to_tx(), tx.status.into_confirmation_time())
                        {
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

        let blocks_at_end = self.recent_blocks()?;

        for block in blocks_at_end {
            if update.insert_checkpoint(block).is_err() {
                return Err(UpdateError::Reorg);
            }
        }

        Ok(wallet_scan)
    }
}
