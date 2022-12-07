mod electrum;

use std::fmt::Debug;

use bdk_core::{bitcoin::Transaction, sparse_chain::SparseChain, BlockId, TxHeight};
use bdk_keychain::{KeychainScan, KeychainTracker};
use electrum::ElectrumClient;

use bdk_cli::{
    anyhow::{self, Context, Result},
    clap::{self, Subcommand},
};
use log::debug;

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

fn fetch_transactions<K: Debug + Ord + Clone>(
    new_sparsechain: &SparseChain<TxHeight>,
    client: &ElectrumClient,
    tracker: &KeychainTracker<K, TxHeight>,
) -> Result<Vec<(Transaction, Option<TxHeight>)>> {
    // Changeset of txids, both new and old.
    let txid_changeset = tracker.chain().determine_changeset(new_sparsechain)?.txids;

    // Only filter for txids that are new to us.
    let new_txids = txid_changeset
        .iter()
        .filter_map(|(txid, index)| {
            if !tracker.graph().contains_txid(*txid) {
                Some((txid, index))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    // Remaining of the transactions that only changed in Index
    let existing_txs = txid_changeset
        .iter()
        .filter_map(|(txid, index)| match tracker.graph().tx(*txid) {
            Some(tx) => Some((tx.clone(), *index)),
            // We don't keep the index for `TxNode::Partial`s
            _ => None,
        })
        .collect::<Vec<_>>();

    let new_transactions = client.batch_transaction_get(new_txids.iter().map(|(txid, _)| *txid))?;

    // Add all the transaction, new and existing into scan_update
    let transaction_update = new_transactions
        .into_iter()
        .zip(new_txids.into_iter().map(|(_, index)| *index))
        .chain(existing_txs)
        .collect::<Vec<_>>();

    Ok(transaction_update)
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<ElectrumCommands, _>()?;

    let client = ElectrumClient::new("ssl://electrum.blockstream.info:60002")?;

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

    match electrum_cmd {
        ElectrumCommands::Scan { stop_gap } => {
            let scripts = tracker.txout_index.iter_all_script_pubkeys_by_keychain();

            let mut keychain_scan = KeychainScan::default();

            // Wallet scan returns a sparse chain that contains new All the BlockIds and Txids
            // relevant to the wallet, along with keychain index update if required.
            let (new_sparsechain, keychain_index_update) =
                client.wallet_txid_scan(scripts, Some(stop_gap), tracker.chain().checkpoints())?;

            keychain_scan.last_active_indexes = keychain_index_update;

            // Inserting everything from the new_sparsechain should be okay as duplicate
            // data would be rejected at the time of update application.
            for (height, hash) in new_sparsechain.checkpoints() {
                let _ = keychain_scan.update.insert_checkpoint(BlockId {
                    height: *height,
                    hash: *hash,
                })?;
            }

            // Fetch the new and old transactions to be added in update structure
            for (tx, index) in fetch_transactions(&new_sparsechain, &client, &tracker)? {
                keychain_scan.update.insert_tx(tx, index)?;
            }

            // Apply the full scan update
            let changeset = tracker.determine_changeset(&keychain_scan)?;
            db.append_changeset(&changeset)?;
            tracker.apply_changeset(changeset);
            debug!("sync completed!!")
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

            let mut scan_update = KeychainScan::default();

            // Wallet scan returns a sparse chain that contains new All the BlockIds and Txids
            // relevant to the wallet, along with keychain index update if required.
            let new_sparsechain = client
                .spk_txid_scan(spks, tracker.chain().checkpoints())
                .context("scanning the blockchain")?;

            // Inserting everything from the new_sparsechain should be okay as duplicate
            // data would be rejected at the time of update application.
            for (height, hash) in new_sparsechain.checkpoints() {
                let _ = scan_update.update.insert_checkpoint(BlockId {
                    height: *height,
                    hash: *hash,
                })?;
            }

            // Fetch the new and old transactions to be added in update structure
            for (tx, index) in fetch_transactions(&new_sparsechain, &client, &tracker)? {
                scan_update.update.insert_tx(tx, index)?;
            }

            // Apply the full scan update
            let changeset = tracker.determine_changeset(&scan_update)?;
            db.append_changeset(&changeset)?;
            tracker.apply_changeset(changeset);
            debug!("sync completed!!")
        }
    }
    Ok(())
}
