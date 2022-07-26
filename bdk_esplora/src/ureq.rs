use crate::api::Tx;
use bdk_core::{
    bitcoin::{
        consensus,
        hashes::{hex::ToHex, sha256, Hash},
        BlockHash, Script, Transaction, Txid,
    },
    BlockId, CheckpointCandidate,
};
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
    TipChangeDuringUpdate,
    Deserialization { url: String },
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
            UpdateError::TipChangeDuringUpdate => {
                write!(f, "The blockchain tip changed during the update")
            }
            UpdateError::Deserialization { url } => {
                write!(f, "Failed to deserialize response from {}", url)
            }
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
    ) -> Result<Vec<Tx>, ureq::Error> {
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

    fn is_block_present(&self, block: BlockId) -> Result<bool, ureq::Error> {
        use core::str::FromStr;
        let url = format!("{}/block-height/{}", self.base_url, block.height);
        let response = self.agent.get(&url).call()?;
        Ok(BlockHash::from_str(&response.into_string()?) == Ok(block.hash))
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

    /// Create a new checkpoint with transactions spending from or to the scriptpubkeys in
    /// `scripts`.
    pub fn fetch_new_checkpoint(
        &self,
        mut scripts: impl Iterator<Item = (u32, Script)>,
        stop_gap: usize,
        known_tips: impl Iterator<Item = BlockId>,
    ) -> Result<(Option<u32>, CheckpointCandidate), UpdateError> {
        let mut empty_scripts = 0;
        let mut transactions = vec![];
        let mut last_active_index = None;
        let mut invalidate = None;
        let mut base_tip = None;

        for tip in known_tips {
            if self.is_block_present(tip)? {
                base_tip = Some(tip);
                break;
            } else {
                invalidate = Some(tip);
            }
        }

        let new_tip = self.tip()?;

        loop {
            let handles = (0..self.parallel_requests)
                .filter_map(|_| {
                    let (index, script) = scripts.next()?;
                    let client = self.clone();
                    Some(std::thread::spawn(move || {
                        let mut related_txs: Vec<Tx> = client._scripthash_txs(&script, None)?;

                        let n_confirmed =
                            related_txs.iter().filter(|tx| tx.status.confirmed).count();
                        // esplora pages on 25 confirmed transactions. If there's 25 or more we
                        // keep requesting to see if there's more.
                        if n_confirmed >= 25 {
                            loop {
                                let new_related_txs: Vec<Tx> = client._scripthash_txs(
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
                    transactions.push((tx.to_tx(), tx.status.to_block_time()))
                }
            }

            if n_handles == 0 || empty_scripts >= stop_gap {
                break;
            }
        }

        if self.tip_hash()? != new_tip.hash {
            return Err(UpdateError::TipChangeDuringUpdate);
        }

        let update = CheckpointCandidate {
            transactions,
            base_tip,
            invalidate,
            new_tip,
        };

        Ok((last_active_index, update))
    }
}
