mod electrum;
use bdk_core::bitcoin::Network;
use bdk_keychain::KeychainChangeSet;
use electrum::ElectrumClient;
use std::fmt::Debug;

use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Subcommand},
};

use electrum_client::ElectrumApi;

#[derive(Subcommand, Debug, Clone)]
enum ElectrumCommands {
    /// Scans the addresses in the wallet using esplora API.
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: usize,
    },
    /// Scans particular addresses using esplora API
    Sync {
        /// Scan all the unused addresses
        #[clap(long)]
        unused: bool,
        /// Scan the script addresses that have unspent outputs
        #[clap(long)]
        unspent: bool,
        /// Scan every address that you have derived
        #[clap(long)]
        all: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<ElectrumCommands, _>()?;

    let electrum_url = match args.network {
        Network::Bitcoin => "ssl://electrum.blockstream.info:50002",
        Network::Testnet => "ssl://electrum.blockstream.info:60002",
        Network::Regtest => "ssl://localhost:60401",
        // TODO: Find a electrum signet endpoint
        Network::Signet => return Err(anyhow::anyhow!("Signet nor supported for Electrum")),
    };

    let client = ElectrumClient::new(electrum_url)?;

    let electrum_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(electrum_cmd) => electrum_cmd,
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

    let mut changeset = KeychainChangeSet::default();

    let update_chain = match electrum_cmd {
        ElectrumCommands::Scan { stop_gap } => {
            let scripts = tracker.txout_index.iter_all_script_pubkeys_by_keychain();

            // Wallet scan returns a sparse chain that contains new All the BlockIds and Txids
            // relevant to the wallet, along with keychain index update if required.
            let (new_sparsechain, keychain_index_update) =
                client.wallet_txid_scan(scripts, Some(stop_gap), tracker.chain().checkpoints())?;

            tracker.txout_index.store_all_up_to(&keychain_index_update);
            changeset.derivation_indices = tracker.txout_index.derivation_indices();

            new_sparsechain
        }
        ElectrumCommands::Sync {
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

            // Wallet scan returns a sparse chain that contains new All the BlockIds and Txids
            // relevant to the wallet, along with keychain index update if required.
            let new_sparsechain = client
                .spk_txid_scan(spks, tracker.chain().checkpoints())
                .context("scanning the blockchain")?;

            new_sparsechain
        }
    };

    // Changeset of txids, both new and old.
    let sparse_chain_changeset = tracker.chain().determine_changeset(&update_chain)?;

    // Only filter for txids that are new to us.
    let new_txids: Vec<_> = sparse_chain_changeset
        .txids
        .iter()
        // filter the transactions being removed
        .filter_map(|(txid, index)| Some((txid, (*index)?)))
        // filter the transactions that have moved but we already have stored
        .filter(|(txid, index)| Some(index) != tracker.chain().tx_index(**txid))
        .collect();

    let new_txs = client.batch_transaction_get(new_txids.iter().map(|(txid, _)| *txid))?;
    changeset.chain_graph.chain = sparse_chain_changeset;
    changeset.chain_graph.graph.tx.extend(new_txs);

    // Apply the full scan update
    db.append_changeset(&changeset)?;
    tracker.apply_changeset(changeset);

    Ok(())
}
