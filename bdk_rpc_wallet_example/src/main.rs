use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Subcommand},
};
use bitcoincore_rpc::{
    bitcoincore_rpc_json::{
        ImportMultiOptions, ImportMultiRequest, ImportMultiRequestScriptPubkey,
        ImportMultiRescanSince,
    },
    Auth, Client, RpcApi,
};

use bdk_chain::{
    bitcoin::{consensus::deserialize, Script, Transaction},
    keychain::KeychainScan,
};

use bdk_rpc_wallet_example::{RpcClient, RpcError};
use serde_json::Value;

use std::fmt::Debug;

#[derive(Subcommand, Debug, Clone)]
enum RpcCommands {
    /// Scans for transactions related spks in the tracker
    Scan,
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

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<RpcCommands, _>()?;

    let client = {
        let rpc_url = "127.0.0.1:18443".to_string();
        // Change the auth below to match your node config before running the example
        let rpc_auth = Auth::UserPass("user".to_string(), "password".to_string());
        let tracker = tracker.lock().unwrap();
        let scripts = tracker
            .txout_index
            .inner()
            .all_spks()
            .iter()
            .map(|(_, scripts)| scripts);
        init_for_tracker("main_wallet".to_string(), rpc_url, rpc_auth, scripts)?
    };

    let rpc_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(rpc_cmd) => rpc_cmd,
        general_command => {
            return bdk_cli::handle_commands(
                general_command,
                client,
                &mut tracker,
                &mut db,
                args.network,
                &keymap,
            )
        }
    };

    match rpc_cmd {
        RpcCommands::Scan => {
            let mut tracker = tracker.lock().unwrap();

            // Get the initial update sparsechain. This contains all the new txids to be added.
            let update_chain = client.wallet_scan(tracker.chain().checkpoints())?;

            // Find the full transactions for newly found txids
            let new_txids = tracker
                .chain()
                .changeset_additions(&tracker.chain().determine_changeset(&update_chain)?)
                .collect::<Vec<_>>();

            let new_txs = new_txids
                .iter()
                .map(|txid| {
                    let tx_data = client.get_transaction(&txid, Some(true))?.hex;
                    let tx: Transaction = deserialize(&tx_data)?;
                    Ok(tx)
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()?;

            // Create the final changeset
            let change_set = {
                let chaingraph = tracker
                    .chain_graph()
                    .inflate_update(update_chain, new_txs)
                    .context("inflating changeset")?;

                let keychainscan = KeychainScan {
                    update: chaingraph.clone(),
                    last_active_indices: Default::default(),
                };

                tracker.determine_changeset(&keychainscan)?
            };

            // Apply changeset to tracker and db
            let mut db = db.lock().unwrap();
            db.append_changeset(&change_set)?;
            tracker.apply_changeset(change_set);
        }
    }
    Ok(())
}
