mod rpc;

use bdk_file_store::KeychainStore;

use bdk_cli::{handle_commands, Args, Commands, Keychain, Parser, Result};
use bitcoincore_rpc::RpcApi;
use log::debug;

use bdk_core::{
    bitcoin::{consensus::deserialize, secp256k1::Secp256k1, Transaction},
    TxHeight,
};

use bdk_keychain::{
    miniscript::{Descriptor, DescriptorPublicKey},
    KeychainTracker,
};
use rpc::{RpcClient, RpcConfig};

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;

    let mut tracker = KeychainTracker::default();
    tracker
        .txout_index
        .add_keychain(Keychain::External, descriptor);

    let internal = args
        .change_descriptor
        .map(|descriptor| Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &descriptor))
        .transpose()?;

    let change_keychain = if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        tracker
            .txout_index
            .add_keychain(Keychain::Internal, internal_descriptor);
        Keychain::Internal
    } else {
        Keychain::External
    };

    // Sync from DB first
    let mut db = KeychainStore::<Keychain, TxHeight>::load(args.db_dir.as_path(), &mut tracker)?;

    // The connect with RPC backend
    // This will import all the tracker scripts into core wallet.
    let rpc_url = "127.0.0.1:18443".to_string();
    let rpc_auth = ("user".to_string(), "password".to_string());
    let config = RpcConfig::new(rpc_url, rpc_auth, args.network)?;
    let client = RpcClient::init_for_tracker(&config, &tracker.txout_index)?;

    match args.command {
        Commands::Scan => {
            let last_known_height = tracker
                .chain()
                .checkpoints()
                .iter()
                .last()
                .map(|(&ht, _)| ht);

            let mut keychain_scan = client.wallet_scan(last_known_height)?;

            // Update `keychain_scan` with missing transactions
            let txid_changeset = tracker
                .chain_graph()
                .determine_changeset(&keychain_scan.update)?
                .chain
                .txids
                .into_iter()
                .filter(|item| !tracker.graph().contains_txid(&item.0))
                .collect::<Vec<_>>();

            for (txid, index) in txid_changeset {
                let tx_data = client.get_transaction(&txid, Some(true))?.hex;
                let tx: Transaction = deserialize(&tx_data)?;
                let txid = tx.txid();
                let index = index.expect("Its a new transaction, so new index expected");
                if keychain_scan.update.insert_tx(tx, index)? {
                    debug!("added new wallet tx {} in the TxGraph", txid);
                }
            }

            // Apply the full scan update
            let changeset = tracker.determine_changeset(&keychain_scan)?;
            db.append_changeset(&changeset)?;
            tracker.apply_changeset(changeset);
            debug!("sync completed!!")
        }

        // For everything else run handler
        _ => handle_commands(
            args.command,
            client,
            &mut tracker,
            &mut db,
            args.network,
            &keymap,
            &Keychain::External,
            &change_keychain,
        )?,
    }
    Ok(())
}
