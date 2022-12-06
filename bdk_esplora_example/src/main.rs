mod esplora;
use crate::esplora::Client;
use bdk_core::bitcoin::Network;

use std::io::{self, Write};

const DEFAULT_PARALLEL_REQUESTS: u8 = 5;
use bdk_cli::anyhow::{self, Context};

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut keychain_tracker, mut db) = bdk_cli::init()?;
    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3000",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let client = Client::new(esplora_url, DEFAULT_PARALLEL_REQUESTS)?;

    match args.command {
        bdk_cli::Commands::Scan { stop_gap } => {
            let spk_iterators = keychain_tracker
                .txout_index
                .iter_all_script_pubkeys_by_keychain()
                .into_iter()
                .map(|(keychain, iter)| {
                    let mut first = true;
                    (
                        keychain,
                        iter.inspect(move |(i, _)| {
                            if first {
                                eprint!("\nscanning {}: ", keychain);
                                first = false;
                            }

                            eprint!("{} ", i);
                            let _ = io::stdout().flush();
                        }),
                    )
                })
                .collect();

            let local_chain = keychain_tracker.chain().checkpoints().clone();

            let wallet_scan = client
                .wallet_scan(spk_iterators, &local_chain, Some(stop_gap))
                .context("scanning the blockchain")?;
            eprintln!();

            let changeset = keychain_tracker.determine_changeset(&wallet_scan)?;
            db.append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
        }
        bdk_cli::Commands::Sync {
            mut unused,
            mut unspent,
            all,
        } => {
            let txout_index = &keychain_tracker.txout_index;
            if !(all || unused || unspent) {
                unused = true;
                unspent = true;
            } else if all {
                unused = false;
                unspent = false
            }
            let mut spks: Box<dyn Iterator<Item = bdk_core::bitcoin::Script>> =
                Box::new(core::iter::empty());

            if all {
                spks = Box::new(spks.chain(txout_index.script_pubkeys().iter().map(
                    |(index, script)| {
                        eprintln!("scanning {:?}", index);
                        script.clone()
                    },
                )));
            }

            if unused {
                spks = Box::new(spks.chain(txout_index.iter_unused().map(|(index, script)| {
                    eprintln!("Checking if address at {:?} has been used", index);
                    script.clone()
                })));
            }

            if unspent {
                spks = Box::new(spks.chain(keychain_tracker.utxos().map(|(_index, ftxout)| {
                    eprintln!("checking if {} has been spent", ftxout.outpoint);
                    ftxout.txout.script_pubkey
                })));
            }

            let local_chain = keychain_tracker.chain().checkpoints().clone();
            let scan = client
                .spk_scan(spks, &local_chain, None)
                .context("scanning the blockchain")?;

            let changeset = keychain_tracker
                .chain_graph()
                .determine_changeset(&scan)?
                .into();
            db.append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
        }
        // For everything else run handler
        _ => bdk_cli::handle_commands(
            args.command,
            client,
            &mut keychain_tracker,
            &mut db,
            args.network,
            &keymap,
        )?,
    }
    Ok(())
}
