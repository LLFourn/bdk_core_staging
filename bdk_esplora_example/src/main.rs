mod esplora;
use crate::esplora::Client;
use bdk_chain::bitcoin::Network;

use std::io::{self, Write};

const DEFAULT_PARALLEL_REQUESTS: u8 = 5;
use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Subcommand},
};

#[derive(Subcommand, Debug, Clone)]
enum EsploraCommands {
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
    let (args, keymap, mut keychain_tracker, mut db) = bdk_cli::init::<EsploraCommands, _>()?;
    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3000",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let client = Client::new(esplora_url, DEFAULT_PARALLEL_REQUESTS)?;

    let esplora_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(esplora_cmd) => esplora_cmd,
        general_command => {
            return bdk_cli::handle_commands(
                general_command,
                client,
                &mut keychain_tracker,
                &mut db,
                args.network,
                &keymap,
            )
        }
    };

    match esplora_cmd {
        EsploraCommands::Scan { stop_gap } => {
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
            keychain_tracker
                .apply_changeset(changeset)
                .expect("it was just generated");
        }
        EsploraCommands::Sync {
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
            let mut spks: Box<dyn Iterator<Item = bdk_chain::bitcoin::Script>> =
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
                spks = Box::new(spks.chain(txout_index.unused(..).map(|(index, script)| {
                    eprintln!("Checking if address at {:?} has been used", index);
                    script.clone()
                })));
            }

            if unspent {
                spks = Box::new(spks.chain(keychain_tracker.full_utxos().map(
                    |(_index, ftxout)| {
                        eprintln!("checking if {} has been spent", ftxout.outpoint);
                        ftxout.txout.script_pubkey
                    },
                )));
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
            keychain_tracker
                .apply_changeset(changeset)
                .expect("it was just generated");
        } // For everything else run handler
    }

    Ok(())
}
