mod rpc;

use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Subcommand},
};
use bitcoincore_rpc::RpcApi;

use bdk_chain::{
    bitcoin::{consensus::deserialize, Transaction},
    keychain::KeychainChangeSet,
};

use rpc::{RpcClient, RpcConfig};

#[derive(Subcommand, Debug, Clone)]
enum RpcCommands {
    /// Scans for transactions related spks in the tracker
    Scan,
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<RpcCommands, _>()?;

    let rpc_url = "127.0.0.1:18443".to_string();
    let rpc_auth = ("user".to_string(), "password".to_string());
    let config = RpcConfig::new(rpc_url, rpc_auth, args.network);
    let client = RpcClient::init_for_tracker(&config, &tracker.lock().unwrap().txout_index)?;

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
            let mut keychain_changeset = KeychainChangeSet::default();

            let mut tracker = tracker.lock().unwrap();

            let chain_update = client.wallet_scan(tracker.chain().checkpoints())?;

            let sparsechain_changeset = tracker.chain().determine_changeset(&chain_update)?;

            let new_txids = tracker
                .chain()
                .changeset_additions(&sparsechain_changeset)
                .collect::<Vec<_>>();

            let new_txs = new_txids
                .iter()
                .map(|txid| {
                    let tx_data = client.get_transaction(&txid, Some(true))?.hex;
                    let tx: Transaction = deserialize(&tx_data)?;
                    Ok(tx)
                })
                .collect::<Result<Vec<_>, anyhow::Error>>()?;

            let chaingraph_changeset = tracker
                .chain_graph()
                .inflate_changeset(sparsechain_changeset, new_txs)
                .context("inflating changeset")?;

            keychain_changeset.chain_graph = chaingraph_changeset;

            let mut db = db.lock().unwrap();

            db.append_changeset(&keychain_changeset)?;
            tracker.apply_changeset(keychain_changeset);
        }
    }
    Ok(())
}
