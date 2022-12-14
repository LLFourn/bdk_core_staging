mod electrum;
use bdk_chain::{bitcoin::Network, keychain::KeychainScan};
use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Args, Parser, Subcommand},
};
use electrum::ElectrumClient;
use std::{collections::BTreeMap, fmt::Debug, io, io::Write};

use electrum_client::{Client, ConfigBuilder, ElectrumApi};

#[derive(Args, Debug, Clone)]
struct ElectrumArgs {}

#[derive(Subcommand, Debug, Clone)]
enum ElectrumCommands {
    /// Scans the addresses in the wallet using esplora API.
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: usize,
        #[clap(flatten)]
        scan_option: ScanOption,
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
        #[clap(flatten)]
        scan_option: ScanOption,
    },
}

#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ScanOption {
    /// Set batch size for each script_history call to electrum client
    #[clap(long, default_value = "25")]
    pub batch_size: usize,
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<ElectrumArgs, ElectrumCommands, _>()?;

    let electrum_url = match args.network {
        Network::Bitcoin => "ssl://electrum.blockstream.info:50002",
        Network::Testnet => "ssl://electrum.blockstream.info:60002",
        Network::Regtest => "tcp://localhost:60401",
        Network::Signet => "tcp://signet-electrumx.wakiyamap.dev:50001",
    };
    let config = ConfigBuilder::new()
        .validate_domain(match args.network {
            Network::Bitcoin => true,
            _ => false,
        })
        .build();

    let client = ElectrumClient::new(Client::from_config(electrum_url, config)?)?;

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

    let (chain_update, last_active_indices) = match electrum_cmd {
        ElectrumCommands::Scan {
            stop_gap,
            scan_option,
        } => {
            let (spk_iterators, local_chain) = {
                // Get a short lock on the tracker to get the spks iterators
                // and local chain state
                let tracker = &*tracker.lock().unwrap();
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

            // we scan the spks **wihtout** a lock on the tracker
            let (new_sparsechain, last_active_indices) = client.wallet_txid_scan(
                spk_iterators,
                Some(stop_gap),
                &local_chain,
                scan_option.batch_size,
            )?;

            eprintln!();

            (new_sparsechain, last_active_indices)
        }
        ElectrumCommands::Sync {
            mut unused,
            mut unspent,
            all,
            scan_option,
        } => {
            let (spks, local_chain) = {
                // Get a short lock on the tracker to get the spks we're interested in
                let tracker = &*tracker.lock().unwrap();
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

            // we scan the spks **without** a lock on the tracker
            let new_sparsechain = client
                .spk_txid_scan(spks, &local_chain, scan_option.batch_size)
                .context("scanning the blockchain")?;

            (new_sparsechain, BTreeMap::default())
        }
    };

    let new_txids = {
        let tracker = &*tracker.lock().unwrap();
        chain_update
            .txids()
            .filter(|(_, txid)| tracker.graph().get_tx(*txid).is_none())
            .map(|&(_, txid)| txid)
            .collect::<Vec<_>>()
    };

    // fetch the missing full transactions **without** a lock on the tracker
    let new_txs = client
        .batch_transaction_get(new_txids.iter())
        .context("fetching full transactions")?;

    {
        // Get a final short lock to apply the changes
        let tracker = &mut *tracker.lock().unwrap();
        let update = tracker
            .chain_graph()
            .inflate_update(chain_update, new_txs)
            .context("inflating update")?;
        let changeset = {
            let keychain_scan = KeychainScan {
                update: update,
                last_active_indices,
            };
            tracker.determine_changeset(&keychain_scan)?
        };
        let db = &mut *db.lock().unwrap();
        db.append_changeset(&changeset)?;
        tracker.apply_changeset(changeset);
    }
    Ok(())
}
