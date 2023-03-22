use std::collections::HashMap;
use std::env;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use bdk_chain::bitcoin::secp256k1::Secp256k1;
use bdk_chain::bitcoin::Script;
use bdk_chain::file_store::KeychainStore;
use bdk_chain::keychain::{KeychainChangeSet, KeychainTracker};
use bdk_chain::miniscript::descriptor::DescriptorSecretKey;
use bdk_chain::miniscript::{Descriptor, DescriptorPublicKey};
use bdk_chain::sparse_chain::SparseChain;
use bdk_chain::{BlockId, TxHeight};
use bdk_rpc_wallet_example::{RpcClient, RpcError};
use bitcoincore_rpc::bitcoincore_rpc_json::{
    ImportMultiOptions, ImportMultiRequest, ImportMultiRequestScriptPubkey, ImportMultiRescanSince,
};
use bitcoincore_rpc::Auth;
use electrsd::bitcoind::{self, BitcoinD};

use electrsd::bitcoind::bitcoincore_rpc::{Client, RpcApi};

use bdk_chain::bitcoin::{Address, Amount, Network, Txid};

use log::{debug, log_enabled, Level};

use bdk_cli::anyhow::{anyhow, Result};

use bdk_cli::Keychain;
use serde_json::Value;

pub struct TestClient {
    pub bitcoind: BitcoinD,
}

impl Deref for TestClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.bitcoind.client
    }
}

impl TestClient {
    pub fn new() -> Self {
        let bitcoind_exe = env::var("BITCOIND_EXE")
            .ok()
            .or(bitcoind::downloaded_exe_path().ok())
            .expect(
                "you should provide env var BITCOIND_EXE or specifiy a bitcoind version feature",
            );

        debug!("launching {}", &bitcoind_exe);

        let mut conf = bitcoind::Conf::default();
        conf.view_stdout = log_enabled!(Level::Debug);
        let bitcoind = BitcoinD::with_conf(bitcoind_exe, &conf).unwrap();

        let node_address = bitcoind.client.get_new_address(None, None).unwrap();
        bitcoind
            .client
            .generate_to_address(101, &node_address)
            .unwrap();

        TestClient { bitcoind }
    }

    pub fn receive(&self, address: Address, amount: Amount, min_conf: Option<u64>) -> Result<Txid> {
        if self.get_balance(None, None)? < amount {
            return Err(anyhow!("Not enough balance in test wallet"));
        }
        let txid = self.send_to_address(&address, amount, None, None, None, None, None, None)?;
        if let Some(num) = min_conf {
            self.generate(num, None);
        } else {
            self.generate(1, None);
        }
        debug!("Sent tx: {}", txid);
        Ok(txid)
    }

    pub fn bump_fee(&mut self, txid: &Txid) -> Result<Txid> {
        let tx = self.get_raw_transaction_info(txid, None)?;
        assert!(
            tx.confirmations.is_none(),
            "Can't bump tx {} because it's already confirmed",
            txid
        );
        let new_txid = self.bump_fee(txid)?;
        debug!("Bumped {}, new txid {}", txid, new_txid);
        Ok(new_txid)
    }

    pub fn generate(&self, num_blocks: u64, address: Option<Address>) {
        let address = address.unwrap_or_else(|| self.get_new_address(None, None).unwrap());
        let hashes = self.generate_to_address(num_blocks, &address).unwrap();
        let best_hash = hashes.last().unwrap();
        let height = self.get_block_info(best_hash).unwrap().height;

        debug!("Generated blocks to new height {}", height);
    }

    pub fn invalidate(&self, num_blocks: u64) -> Result<()> {
        let best_hash = self.get_best_block_hash()?;
        let initial_height = self.get_block_info(&best_hash)?.height;

        let mut to_invalidate = best_hash;
        for i in 1..=num_blocks {
            debug!(
                "Invalidating block {}/{} ({})",
                i, num_blocks, to_invalidate
            );

            self.invalidate_block(&to_invalidate)?;
            to_invalidate = self.get_best_block_hash()?;
        }

        debug!(
            "Invalidated {} blocks to new height of {}",
            num_blocks,
            initial_height - num_blocks as usize
        );

        Ok(())
    }

    pub fn reorg(&self, num_blocks: u64) -> Result<()> {
        self.invalidate(num_blocks)?;
        self.generate(num_blocks, None);
        Ok(())
    }

    pub fn get_node_address(&self) -> Result<Address> {
        Ok(Address::from_str(
            &self.get_new_address(None, None)?.to_string(),
        )?)
    }

    pub fn get_config(&self) -> (String, Auth) {
        (
            self.bitcoind.params.rpc_socket.to_string(),
            bitcoincore_rpc::Auth::CookieFile(self.bitcoind.params.cookie_file.clone()),
        )
    }

    pub fn get_update_sparsechain<K>(
        &self,
        tracker: &mut KeychainTracker<K, TxHeight>,
    ) -> Result<SparseChain>
    where
        K: Ord + Clone + core::fmt::Debug,
        KeychainChangeSet<K, TxHeight>: serde::Serialize + serde::de::DeserializeOwned,
    {
        let (url, auth) = self.get_config();

        let scripts = tracker
            .txout_index
            .inner()
            .all_spks()
            .iter()
            .map(|(_, script)| script);

        let rpc_client = init_for_tracker("main_wallet".to_string(), url, auth, scripts)?;

        Ok(rpc_client.wallet_scan(tracker.chain().checkpoints())?)
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

/// Initialize [`RpcClient`] from a [`RpcConfig`] and [`KeychainTxOutIndex`].
/// This creates a wallet inside bitcoin core, with a deterministic name
/// derived from the hash of descriptors within the [`KeychainTxOutIndex`].
/// Then imports all the spks from the index into the newly created core wallet.
pub fn init_for_tracker<'a>(
    wallet_name: String,
    url: String,
    auth: Auth,
    scripts: impl Iterator<Item = &'a Script>,
) -> Result<RpcClient, RpcError> {
    let wallet_url = format!("{}/wallet/{}", url, wallet_name);
    let client = Client::new(wallet_url.as_str(), auth.clone().into())?;
    let rpc_version = client.version()?;
    println!("rpc connection established. Core version : {}", rpc_version);
    println!("connected to '{}' with auth: {:?}", wallet_url, auth);

    if client.list_wallets()?.contains(&wallet_name.to_string()) {
        println!("wallet already loaded: {}", wallet_name);
    } else if list_wallet_dirs(&client)?.contains(&wallet_name.to_string()) {
        println!("wallet wasn't loaded. Loading wallet : {}", wallet_name);
        client.load_wallet(&wallet_name)?;
        println!("wallet loaded: {}", wallet_name);
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
        println!("wallet created: {}", wallet_name);
    }

    import_multi(&client, scripts)?;

    Ok(RpcClient { client })
}

pub fn init() -> Result<(
    HashMap<DescriptorPublicKey, DescriptorSecretKey>,
    KeychainTracker<Keychain, TxHeight>,
    KeychainStore<Keychain, TxHeight>,
)>
where
    KeychainChangeSet<Keychain, TxHeight>: serde::Serialize + serde::de::DeserializeOwned,
{
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)")?;

    let mut tracker = KeychainTracker::default();
    tracker.set_checkpoint_limit(None);

    tracker
        .txout_index
        .add_keychain(Keychain::External, descriptor);

    let (internal_descriptor, internal_keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp,  "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)")?;

    keymap.extend(internal_keymap);
    tracker
        .txout_index
        .add_keychain(Keychain::Internal, internal_descriptor);

    tracker.txout_index.set_lookahead_for_all(10);
    tracker.set_checkpoint_limit(Some(50));

    // Do not reload a previous test db file. Delete previous file if it exists.
    if Path::new("./test-db").exists() {
        std::fs::remove_file("./test-db").unwrap();
    }

    let mut db = KeychainStore::<Keychain, TxHeight>::new_from_path("./test-db")?;

    if let Err(e) = db.load_into_keychain_tracker(&mut tracker) {
        match tracker.chain().latest_checkpoint()  {
            Some(checkpoint) => eprintln!("Failed to load all changesets from {}. Last checkpoint was at height {}. Error: {}", "./test-db", checkpoint.height, e),
            None => eprintln!("Failed to load any checkpoints from {}: {}", "./test-db", e),

        }
        eprintln!("⚠ Consider running a rescan of chain data.");
    }

    Ok((keymap, tracker, db))
}

#[test]
fn basic_sync_test() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .is_test(true)
        .try_init()
        .unwrap();

    let (_, mut tracker, mut db) = init().unwrap();
    let bitcoind = TestClient::new();

    // Get a new address
    let ((index, script), additions) = tracker.txout_index.next_unused_spk(&Keychain::External);
    assert_eq!(index, 0);
    db.append_changeset(&additions.into()).unwrap();

    // Receive funds into the address
    let txid = bitcoind
        .receive(
            Address::from_script(&script, Network::Regtest).unwrap(),
            Amount::ONE_BTC,
            None,
        )
        .unwrap();
    let update_chain = bitcoind.get_update_sparsechain(&mut tracker).unwrap();

    let blockhash = bitcoind.get_best_block_hash().unwrap();

    let checkpoint = update_chain.latest_checkpoint().unwrap();
    assert_eq!(
        checkpoint,
        BlockId {
            height: 102,
            hash: blockhash
        }
    );

    let txids = update_chain.txids().collect::<Vec<_>>();
    assert_eq!(txids, vec![&(TxHeight::Confirmed(102), txid)]);

    std::fs::remove_file("./test-db").unwrap();
}

#[test]
fn basic_reorg() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .is_test(true)
        .try_init()
        .unwrap();

    let (_, mut tracker, mut db) = init().unwrap();
    let bitcoind = TestClient::new();

    // Get a new address
    let ((index, script), additions) = tracker.txout_index.next_unused_spk(&Keychain::External);
    assert_eq!(index, 0);
    db.append_changeset(&additions.into()).unwrap();

    // Receive funds in that address
    let txid = bitcoind
        .receive(
            Address::from_script(&script, Network::Regtest).unwrap(),
            Amount::ONE_BTC,
            None,
        )
        .unwrap();
    let update_chain = bitcoind.get_update_sparsechain(&mut tracker).unwrap();

    let blockhash = bitcoind.get_best_block_hash().unwrap();
    let checkpoint = update_chain.latest_checkpoint().unwrap();
    assert_eq!(
        checkpoint,
        BlockId {
            height: 102,
            hash: blockhash
        }
    );
    let txids = update_chain.txids().collect::<Vec<_>>();
    assert_eq!(txids, vec![&(TxHeight::Confirmed(102), txid)]);

    // Reorg last 5 blocks
    bitcoind.reorg(5).unwrap();

    let update_chain = bitcoind.get_update_sparsechain(&mut tracker).unwrap();

    let blockhash = bitcoind.get_best_block_hash().unwrap();
    let checkpoint = update_chain.latest_checkpoint().unwrap();
    assert_eq!(
        checkpoint,
        BlockId {
            height: 102,
            hash: blockhash
        }
    );
    let txids = update_chain.txids().collect::<Vec<_>>();
    assert_eq!(txids, Vec::<&(TxHeight, Txid)>::new());

    std::fs::remove_file("./test-db").unwrap();
}
