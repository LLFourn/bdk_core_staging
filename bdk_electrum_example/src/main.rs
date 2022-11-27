mod electrum;

use bdk_file_store::KeychainStore;
use electrum::ElectrumClient;

use bdk_cli::{handle_commands, Args, Commands, Context, Keychain, Parser, Result};
use log::debug;

use bdk_core::{bitcoin::secp256k1::Secp256k1, TxHeight};

use electrum_client::ElectrumApi;

use bdk_keychain::{
    miniscript::{Descriptor, DescriptorPublicKey},
    KeychainTracker,
};

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

    let client = ElectrumClient::new("ssl://electrum.blockstream.info:60002")?;

    let mut db = KeychainStore::<Keychain, TxHeight>::load(args.db_dir.as_path(), &mut tracker)?;

    match args.command {
        Commands::Scan => {
            let last_known_height = tracker
                .chain()
                .checkpoints()
                .iter()
                .last()
                .map(|(&ht, _)| ht);

            let keychain_scripts = tracker.txout_index.iter_all_spks();

            let mut keychain_scan =
                client.wallet_txid_scan(keychain_scripts, Some(10), last_known_height)?;

            // Update `keychain_scan` with missing transactions
            let txid_changeset = tracker
                .chain_graph()
                .determine_changeset(&keychain_scan.update)?
                .chain
                .txids
                .into_iter()
                .filter(|item| !tracker.graph().contains_txid(&item.0))
                .collect::<Vec<_>>();

            let new_transactions =
                client.batch_transaction_get(txid_changeset.iter().map(|(txid, _)| txid))?;

            for (tx, index) in new_transactions.into_iter().zip(
                txid_changeset
                    .iter()
                    .map(|(_, index)| index.expect("expect")),
            ) {
                keychain_scan.update.insert_tx(tx, index)?;
            }

            // Apply the full scan update
            let changeset = tracker.determine_changeset(&keychain_scan)?;
            db.append_changeset(&changeset)?;
            tracker.apply_changeset(changeset);
            debug!("sync completed!!")
        }
        Commands::Sync {
            mut unused,
            mut unspent,
            all,
        } => {
            let txout_index = &tracker.txout_index;
            if !(all || unused || unspent) {
                unused = true;
                unspent = true;
            } else if all {
                unused = false;
                unspent = false
            }
            let mut spks: Box<dyn Iterator<Item = bdk_core::bitcoin::Script>> =
                Box::new(core::iter::empty());
            if unused {
                spks = Box::new(spks.chain(txout_index.iter_unused().map(|(index, script)| {
                    eprintln!("Checking if address at {:?} has been used", index);
                    script.clone()
                })));
            }

            if all {
                spks = Box::new(spks.chain(txout_index.script_pubkeys().iter().map(
                    |(index, script)| {
                        eprintln!("scanning {:?}", index);
                        script.clone()
                    },
                )));
            }

            if unspent {
                spks = Box::new(spks.chain(tracker.utxos().map(|(_index, ftxout)| {
                    eprintln!("checking if {} has been spent", ftxout.outpoint);
                    ftxout.txout.script_pubkey
                })));
            }

            let last_known_height = tracker
                .chain()
                .checkpoints()
                .iter()
                .last()
                .map(|(&ht, _)| ht);

            let mut chaingraph_update = client
                .spk_txid_scan(spks, last_known_height)
                .context("scanning the blockchain")?;

            // Update `chaingraph_update` with missing transaction data.
            let txid_changeset = tracker
                .chain_graph()
                .determine_changeset(&chaingraph_update)?
                .chain
                .txids
                .into_iter()
                .filter(|item| !tracker.graph().contains_txid(&item.0))
                .collect::<Vec<_>>();

            // Fetch and add these new transactions into update candidate
            let new_transactions =
                client.batch_transaction_get(txid_changeset.iter().map(|(txid, _)| txid))?;

            for (tx, index) in new_transactions.into_iter().zip(
                txid_changeset
                    .iter()
                    .map(|(_, index)| index.expect("expect")),
            ) {
                chaingraph_update.insert_tx(tx, index)?;
            }

            // Apply `chaingraph_update`.
            let changeset = tracker
                .chain_graph()
                .determine_changeset(&chaingraph_update)?
                .into();
            db.append_changeset(&changeset)?;
            tracker.apply_changeset(changeset);

            debug!("sync completed!!");
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
