use bdk_cli::{anyhow, Broadcast, Result};
use bdk_core::{
    bitcoin::{
        hashes::sha256::Hash as sha256Hash,
        hashes::{hex::ToHex, Hash},
        Network, Script, Txid,
    },
    BlockId, TxHeight,
};
use bdk_keychain::{KeychainScan, KeychainTxOutIndex};
use bitcoincore_rpc::{
    bitcoincore_rpc_json::{
        ImportMultiOptions, ImportMultiRequest, ImportMultiRequestScriptPubkey,
        ImportMultiRescanSince,
    },
    Auth, Client, RpcApi,
};
use log::debug;
use serde_json::Value;

use std::{collections::HashSet, fmt::Debug, ops::Deref};

const REORG_DETECTION_DEPTH: u32 = 100;

/// A Structure containing RPC connection information
pub struct RpcConfig {
    url: String,
    auth: Auth,
    network: Network,
}

impl RpcConfig {
    //TODO : Add cookie authentication
    pub fn new(url: String, user_pass: (String, String), network: Network) -> Result<Self> {
        let auth = Auth::UserPass(user_pass.0, user_pass.1);
        Ok(Self { url, auth, network })
    }
}

/// Return list of bitcoin core wallets
fn list_wallet_dir(client: &Client) -> Result<Vec<String>> {
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

/// Create unique wallet name from [`KeychainTracker`].
pub fn derive_name_from_keychainindex<K: Debug + Clone + Ord>(
    txout_index: &KeychainTxOutIndex<K>,
) -> String {
    let data = txout_index
        .keychains(..)
        .fold(Vec::new(), |mut accum, item| {
            let mut bytes = item.1.to_string().as_bytes().to_vec();
            accum.append(&mut bytes);
            return accum;
        });
    sha256Hash::hash(&data[..]).to_hex()[0..10].to_string()
}

/// Import Script Pubkeys into bitcoin core wallet with optional label.
pub fn import_multi<'a>(client: &Client, scripts: impl Iterator<Item = &'a Script>) -> Result<()> {
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
            return Err(anyhow!("Importmulti Error: {:?}", err));
        }
    }
    Ok(())
}

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
    /// Initialize a [`Client`] with given wallet name and connection information.
    pub fn init_for_tracker<K: Debug + Clone + Ord>(
        config: &RpcConfig,
        txout_index: &KeychainTxOutIndex<K>,
    ) -> Result<Self> {
        let wallet_name = derive_name_from_keychainindex(txout_index);
        let wallet_url = format!("{}/wallet/{}", config.url, wallet_name);
        let client = Client::new(wallet_url.as_str(), config.auth.clone().into())?;
        let rpc_version = client.version()?;
        debug!("rpc connection established. Core version : {}", rpc_version);
        debug!("connected to '{}' with auth: {:?}", wallet_url, config.auth);

        if client.list_wallets()?.contains(&wallet_name.to_string()) {
            debug!("wallet already loaded: {}", wallet_name);
        } else if list_wallet_dir(&client)?.contains(&wallet_name.to_string()) {
            debug!("wallet wasn't loaded. Loading wallet : {}", wallet_name);
            client.load_wallet(&wallet_name)?;
            debug!("wallet loaded: {}", wallet_name);
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

            debug!("wallet created: {}", wallet_name);
        }

        let blockchain_info = client.get_blockchain_info()?;
        let network = match blockchain_info.chain.as_str() {
            "main" => Network::Bitcoin,
            "test" => Network::Testnet,
            "regtest" => Network::Regtest,
            "signet" => Network::Signet,
            _ => return Err(anyhow!("Invalid Network string")),
        };
        if network != config.network {
            return Err(anyhow!("Invalid Network"));
        }

        let scripts = txout_index
            .iter_all_spks()
            .into_values()
            .flatten()
            .map(|(_, script)| script)
            .collect::<Vec<_>>();

        let _ = import_multi(&client, scripts.iter())?;

        Ok(Self { client })
    }

    fn get_height(&self) -> Result<u32> {
        Ok(self.client.get_blockchain_info().map(|i| i.blocks as u32)?)
    }

    /// Scan for a keychain tracker, and create an initial [`KeychainScan`] candidate update.
    /// This update will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and update the [`KeychainScan`] before applying it.
    pub fn wallet_scan<K: Ord + Clone>(
        &self,
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

        let block_ids = (check_from..=current_height)
            .map(|height| {
                let hash = self.get_block_hash(height as u64)?;
                Ok(BlockId { height, hash })
            })
            .collect::<Result<Vec<_>>>()?;

        for block in block_ids {
            keychain_scan.update.chain.insert_checkpoint(block)?;
        }

        // Fill the transactions
        let page_size = 1000; // Core has 1000 page size limit
        let mut page = 0;

        let _ = self.rescan_blockchain(Some(check_from as usize), None);

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
                        Some(height) => TxHeight::Confirmed(height),
                        None => TxHeight::Unconfirmed,
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
            keychain_scan.update.chain.insert_tx(txid, index)?;
        }

        Ok(keychain_scan)
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
    fn broadcast(&self, tx: &bdk_core::bitcoin::Transaction) -> Result<()> {
        let _ = self.client.send_raw_transaction(tx)?;
        Ok(())
    }
}
