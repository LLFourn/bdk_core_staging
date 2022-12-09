mod electrum;

use electrum::ElectrumClient;

use bdk_cli::{
    anyhow::{self, Context},
    handle_commands, Commands,
};
use log::debug;

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init()?;

    let client = ElectrumClient::new("ssl://electrum.blockstream.info:60002")?;

    match args.command {
        Commands::Scan { stop_gap } => {
            let last_known_height = tracker
                .chain()
                .checkpoints()
                .iter()
                .last()
                .map(|(&ht, _)| ht);

            let scripts = tracker.txout_index.iter_all_script_pubkeys_by_keychain();

            let mut keychain_scan =
                client.wallet_txid_scan(scripts, Some(stop_gap), last_known_height)?;

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
        )?,
    }
    Ok(())
}
