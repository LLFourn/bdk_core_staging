use bdk_chain::{
    bitcoin::{
        hashes::sha256::Hash as sha256Hash,
        hashes::{hex::ToHex, Hash},
        BlockHash, Network, Script, Transaction, Txid,
    },
    keychain::KeychainTxOutIndex,
    sparse_chain::{self, SparseChain},
    BlockId, TxHeight,
};
use bdk_cli::Broadcast;
use bitcoincore_rpc::{
    bitcoincore_rpc_json::{
        ImportMultiOptions, ImportMultiRequest, ImportMultiRequestScriptPubkey,
        ImportMultiRescanSince, ImportMultiResultError,
    },
    Auth, Client, RpcApi,
};
use serde_json::Value;

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Debug,
    ops::Deref,
};

/// Bitcoin Core RPC related errors.
#[derive(Debug)]
pub enum RpcError {
    Client(bitcoincore_rpc::Error),
    Reorg,
    ImportMulti(ImportMultiResultError),
    General(String),
}

impl core::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RpcError::Client(e) => write!(f, "{}", e),
            RpcError::ImportMulti(e) => write!(f, "{:?}", e),
            RpcError::Reorg => write!(
                f,
                "Reorg detected at sync time. Please run the sync call again"
            ),
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

/// A Structure containing RPC connection information
pub struct RpcConfig {
    url: String,
    auth: Auth,
    network: Network,
}

impl RpcConfig {
    //TODO : Add cookie authentication
    pub fn new(url: String, user_pass: (String, String), network: Network) -> Self {
        let auth = Auth::UserPass(user_pass.0, user_pass.1);
        Self { url, auth, network }
    }
}

/// Return list of bitcoin core wallet directories
fn list_wallet_dirs(client: &Client) -> Result<Vec<String>, RpcError> {
    #[derive(serde::Deserialize)]
    struct Name {
        name: String,
    }
    #[derive(serde::Deserialize)]
    struct CallResult {
        wallets: Vec<Name>,
    }

    let result: CallResult = client.call("listwalletdir", &[])?;
    Ok(result.wallets.into_iter().map(|n| n.name).collect())
}

/// Create unique wallet name from [`KeychainTxOutIndex`].
pub fn derive_name_from_keychainindex<K: Debug + Clone + Ord>(
    txout_index: &KeychainTxOutIndex<K>,
) -> String {
    let data = txout_index
        .keychains()
        .iter()
        .fold(Vec::new(), |mut accum, item| {
            let mut bytes = item.1.to_string().as_bytes().to_vec();
            accum.append(&mut bytes);
            return accum;
        });
    sha256Hash::hash(&data[..]).to_hex()[0..10].to_string()
}

/// Import script_pubkeys into bitcoin core wallet.
pub fn import_multi<'a>(
    client: &Client,
    scripts: impl Iterator<Item = &'a Script>,
) -> Result<(), RpcError> {
    let requests = scripts
        .map(|script| ImportMultiRequest {
            timestamp: ImportMultiRescanSince::Now,
            script_pubkey: Some(ImportMultiRequestScriptPubkey::Script(script)),
            watchonly: Some(true),
            ..Default::default()
        })
        .collect::<Vec<_>>();

    let options = ImportMultiOptions {
        rescan: Some(false),
    };
    for import_multi_result in client.import_multi(&requests, Some(&options))? {
        if let Some(err) = import_multi_result.error {
            return Err(err.into());
        }
    }
    Ok(())
}

/// A Bitcoin Core RPC Client struct that can be used to sync the [`KeychainTxOutIndex`]
pub struct RpcClient {
    client: Client,
}

impl Deref for RpcClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl RpcClient {
    /// Initialize [`RpcClient`] from a [`RpcConfig`] and [`KeychainTxOutIndex`].
    /// This creates a wallet inside bitcoin core, with a deterministic name
    /// derived from the hash of descriptors within the [`KeychainTxOutIndex`].
    /// Then imports all the spks from the index into the newly created core wallet.
    pub fn init_for_tracker<K: Debug + Clone + Ord>(
        config: &RpcConfig,
        txout_index: &KeychainTxOutIndex<K>,
    ) -> Result<Self, RpcError> {
        let wallet_name = derive_name_from_keychainindex(txout_index);
        let wallet_url = format!("{}/wallet/{}", config.url, wallet_name);
        let client = Client::new(wallet_url.as_str(), config.auth.clone().into())?;
        let rpc_version = client.version()?;
        println!("rpc connection established. Core version : {}", rpc_version);
        println!("connected to '{}' with auth: {:?}", wallet_url, config.auth);

        if client.list_wallets()?.contains(&wallet_name.to_string()) {
            println!("wallet already loaded: {}", wallet_name);
        } else if list_wallet_dirs(&client)?.contains(&wallet_name.to_string()) {
            println!("wallet wasn't loaded. Loading wallet : {}", wallet_name);
            client.load_wallet(&wallet_name)?;
            println!("wallet loaded: {}", wallet_name);
        } else {
            // pre-0.21 use legacy wallets
            if rpc_version < 210_000 {
                client.create_wallet(&wallet_name, Some(true), None, None, None)?;
            } else {
                // TODO: move back to api call when https://github.com/rust-bitcoin/rust-bitcoincore-rpc/issues/225 is closed
                let args = [
                    Value::String(String::from(&wallet_name)),
                    Value::Bool(true),  // disable_private_keys
                    Value::Bool(false), //blank
                    Value::Null,        // passphrase
                    Value::Bool(false), // avoid reuse
                    Value::Bool(false), // descriptor
                ];
                let _: Value = client.call("createwallet", &args)?;
            }

            println!("wallet created: {}", wallet_name);
        }

        let blockchain_info = client.get_blockchain_info()?;
        let network = match blockchain_info.chain.as_str() {
            "main" => Network::Bitcoin,
            "test" => Network::Testnet,
            "regtest" => Network::Regtest,
            "signet" => Network::Signet,
            _ => return Err(RpcError::General("Invalid Network string".to_string())),
        };
        if network != config.network {
            return Err(RpcError::General("Invalid Network".to_string()));
        }

        let scripts = txout_index
            .scripts_of_all_keychains()
            .into_values()
            .flatten()
            .map(|(_, script)| script)
            .collect::<Vec<_>>();

        let _ = import_multi(&client, scripts.iter())?;

        Ok(Self { client })
    }

    fn get_tip(&self) -> Result<(u64, BlockHash), RpcError> {
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

        // Find local chain block that is still there so our update can connect to the local chain.
        for (&existing_height, &existing_hash) in local_chain.iter().rev() {
            let current_hash = self.get_block_hash(existing_height as u64)?;
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

        let last_scanned_height = local_chain
            .iter()
            .rev()
            .next()
            .map_or(0, |(height, _)| *height as usize);

        // Fetch the transactions
        let page_size = 1000; // Core has 1000 page size limit
        let mut page = 0;

        let _ = self.rescan_blockchain(Some(last_scanned_height), None);

        let mut txids_to_update = Vec::new();

        loop {
            // TODO: List transaction should take in a label.
            let list_tx_result = self
                .list_transactions(None, Some(page_size), Some(page * page_size), Some(true))?
                .iter()
                .filter(|item|
                    // filter out conflicting transactions - only accept transactions that are already
                    // confirmed, or exists in mempool
                    item.info.confirmations > 0 || self.get_mempool_entry(&item.info.txid).is_ok())
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

        // Check for Reorg during the above sync process
        let our_latest = sparse_chain.latest_checkpoint().expect("must exist");
        if our_latest.hash != self.get_block_hash(our_latest.height as u64)? {
            return Err(RpcError::Reorg);
        }

        Ok(sparse_chain)
    }

    // We can't do random script list sync with RPC, without getting the full wallet
    // related transactions first and then trowing stuffs away.

    // Another option is to create one wallet in core for each query set, which already sounds
    // very nasty in my head.
    // TODO: Figure how to handle any subset of scripts from the wallet.

    // pub fn spk_txid_scan(
    //     &self,
    //     spks: impl Iterator<Item = Script>,
    //     last_known_height: Option<u32>,
    // ) -> Result<ChainGraph<TxHeight>> {
    //     let mut dummy_keychains = BTreeMap::new();
    //     dummy_keychains.insert((), spks.enumerate().map(|(i, spk)| (i as u32, spk)));

    //     let wallet_scan = self.wallet_scan(last_known_height)?;

    //     Ok(wallet_scan.update)
    // }
}

impl Broadcast for RpcClient {
    type Error = bitcoincore_rpc::Error;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error> {
        let _ = self.client.send_raw_transaction(tx)?;
        Ok(())
    }
}
