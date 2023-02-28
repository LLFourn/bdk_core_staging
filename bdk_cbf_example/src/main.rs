use bdk_cli::{
    anyhow,
    clap::{self, Args, Subcommand, ValueEnum},
};

use crate::cbf::CbfEvent;
use bdk_chain::chain_graph::ChainGraph;
use bdk_chain::keychain::{DerivationAdditions, KeychainChangeSet};
use bdk_chain::{BlockId, TxHeight};

mod cbf;

#[derive(Args, Debug, Clone)]
struct CbfArgs {
    #[arg(value_enum, default_value_t = Domains::IPv4)]
    domains: Domains,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum Domains {
    IPv4,
    IPv6,
    Both,
}

#[derive(Subcommand, Debug, Clone)]
enum CbfCommands {
    Rescan {
        #[clap(long, default_value = "1000")]
        _batch_size: u32,
    },
    Scan,
    // Used for manual testing as I still haven't implemented Rescan
    Reveal {
        #[clap(long, default_value = "1000")]
        size: u32,
    },
}

fn main() -> anyhow::Result<()> {
    println!("Loading wallet from db...");
    let (args, keymap, mut keychain_tracker, mut db) =
        bdk_cli::init::<CbfArgs, CbfCommands, TxHeight>()?;
    println!("Wallet loaded.");

    let mut client = cbf::CbfClient::new(args.network.into(), args.chain_args.domains)?;

    let cbf_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(cbf_cmd) => cbf_cmd,
        general_cmd => {
            return bdk_cli::handle_commands(
                general_cmd,
                client,
                &mut keychain_tracker,
                &mut db,
                args.network,
                &keymap,
            );
        }
    };

    match cbf_cmd {
        CbfCommands::Reveal { size } => {
            let mut keychain_tracker = keychain_tracker.lock().unwrap();
            let targets = keychain_tracker
                .keychains()
                .into_iter()
                .map(|(k, _)| (*k, size))
                .collect();
            let (_, derivation_additions) = keychain_tracker
                .txout_index
                .reveal_to_target_multi(&targets);
            let changeset = KeychainChangeSet {
                derivation_indices: derivation_additions,
                ..Default::default()
            };
            db.lock().unwrap().append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
            Ok(())
        }
        CbfCommands::Rescan { _batch_size } => {
            todo!("Implement rescan from sync");
            // This function will reveal batch_size scripts and rescan the nakamoto
            // client
            // If we notice, after rescan, that more than `batch_size - epsilon` scripts
            // have been used, we rescan using 2*batch_size, and so on
        }
        CbfCommands::Scan => {
            // indexing logic!
            let mut keychain_tracker = keychain_tracker.lock().unwrap();

            // find scripts!
            let scripts = keychain_tracker
                .txout_index
                .revealed_spks_of_all_keychains()
                .into_values()
                .flatten()
                .map(|(_, spk)| spk.clone());

            let last_checkpoint = keychain_tracker.chain_graph().chain().latest_checkpoint();

            let mut update = ChainGraph::<TxHeight>::default();

            if let Some(last_cp) = last_checkpoint.clone() {
                // Otherwise the chains don't connect :)
                let _ = update.insert_checkpoint(last_cp.clone())?;
            }

            let mut derivation_additions = DerivationAdditions::default();

            client.sync_setup(scripts, last_checkpoint)?;
            loop {
                match client.next_event()? {
                    CbfEvent::BlockMatched(id, txs) => {
                        let _ = update.insert_checkpoint(BlockId {
                            height: id.height as u32,
                            hash: id.hash,
                        })?;

                        for (tx, height) in txs {
                            //keychain_tracker.lock().unwrap().txout_index.pad_all_with_unused(stop_gap);
                            if keychain_tracker.txout_index.is_relevant(&tx) {
                                println!("* adding tx to update: {} @ {}", tx.txid(), height);
                                let _ = update.insert_tx(tx.clone(), height)?;
                            }
                            derivation_additions.append(keychain_tracker.txout_index.scan(&tx));
                        }
                    }
                    CbfEvent::BlockDisconnected(id) => {
                        // TODO: what happens if a block gets disconnected before I process its
                        // filter?
                        let _ = update.invalidate_checkpoints(id.height as u32);
                    }
                    CbfEvent::ChainSynced(h) => {
                        if let Some(BlockId { hash, height }) = h {
                            let _ = update.insert_checkpoint(BlockId {
                                height: height as u32,
                                hash,
                            })?;
                            println!("chain synced @ height {}", height);
                        }
                        break;
                    }
                }
            }

            let changeset = KeychainChangeSet {
                derivation_indices: derivation_additions,
                chain_graph: keychain_tracker
                    .chain_graph()
                    .determine_changeset(&update)?,
            };
            db.lock().unwrap().append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
            Ok(())
        }
    }
}
