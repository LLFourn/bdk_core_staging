mod esplora;
use crate::esplora::Client;
use bdk_chain::bitcoin::Network;

use std::io::{self, Write};

const DEFAULT_PARALLEL_REQUESTS: u8 = 5;
use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Args, Subcommand},
};

#[derive(Args, Debug, Clone)]
struct EsploraUrlArgs {
    #[clap(env = "ESPLORA_URL", long)]
    url: Option<String>,
}

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
    let (args, keymap, keychain_tracker, db) =
        bdk_cli::init::<EsploraUrlArgs, EsploraCommands, _>()?;

    let esplora_url = match &args.chain_args.url {
        Some(url) => url.as_str(),
        None => match args.network {
            Network::Bitcoin => "https://mempool.space/api",
            Network::Testnet => "https://mempool.space/testnet/api",
            Network::Regtest => "http://localhost:3000",
            Network::Signet => "https://mempool.space/signet/api",
        },
    };

    let client = Client::new(esplora_url, DEFAULT_PARALLEL_REQUESTS)?;

    let esplora_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(esplora_cmd) => esplora_cmd,
        general_command => {
            return bdk_cli::handle_commands(
                general_command,
                client,
                &keychain_tracker,
                &db,
                args.network,
                &keymap,
            )
        }
    };

    match esplora_cmd {
        EsploraCommands::Scan { stop_gap } => {
            let (spk_iterators, local_chain) = {
                // Get a short lock on the tracker to get the spks iterators
                // and local chain state
                let tracker = &*keychain_tracker.lock().unwrap();
                let spk_iterators = tracker
                    .txout_index
                    .spks_of_all_keychains()
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

                let local_chain = tracker.chain().checkpoints().clone();
                (spk_iterators, local_chain)
            };

            // we scan the iterators **without** a lock on the tracker
            let wallet_scan = client
                .wallet_scan(spk_iterators, &local_chain, Some(stop_gap))
                .context("scanning the blockchain")?;
            eprintln!();

            {
                // we take a short lock to apply results to tracker and db
                let tracker = &mut *keychain_tracker.lock().unwrap();
                let db = &mut *db.lock().unwrap();
                let changeset = tracker.apply_update(wallet_scan)?;
                db.append_changeset(&changeset)?;
            }
        }
        EsploraCommands::Sync {
            mut unused,
            mut unspent,
            all,
        } => {
            let (spks, local_chain) = {
                // Get a short lock on the tracker to get the spks we're interested in
                let tracker = &*keychain_tracker.lock().unwrap();
                let txout_index = &tracker.txout_index;
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
                    let all_spks = txout_index
                        .all_spks()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<Vec<_>>();
                    spks = Box::new(spks.chain(all_spks.into_iter().map(|(index, script)| {
                        eprintln!("scanning {:?}", index);
                        script
                    })));
                }

                if unused {
                    let unused_spks = txout_index
                        .unused_spks(..)
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<Vec<_>>();
                    spks = Box::new(spks.chain(unused_spks.into_iter().map(|(index, script)| {
                        eprintln!("Checking if address at {:?} has been used", index);
                        script
                    })));
                }

                if unspent {
                    let unspent_txouts = tracker
                        .full_utxos()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect::<Vec<_>>();
                    spks = Box::new(spks.chain(unspent_txouts.into_iter().map(
                        |(_index, ftxout)| {
                            eprintln!("checking if {} has been spent", ftxout.outpoint);
                            ftxout.txout.script_pubkey
                        },
                    )));
                }
                let local_chain = tracker.chain().checkpoints().clone();

                (spks, local_chain)
            };
            // we scan the desired spks **without** a lock on the tracker
            let scan = client
                .spk_scan(spks, &local_chain, None)
                .context("scanning the blockchain")?;

            {
                // we take a short lock to apply the results to the tracker and db
                let tracker = &mut *keychain_tracker.lock().unwrap();
                let changeset = tracker.apply_update(scan.into())?;
                let db = &mut *db.lock().unwrap();
                db.append_changeset(&changeset)?;
            }
        }
    }

    Ok(())
}
