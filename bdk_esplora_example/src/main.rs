use bdk_chain::{bitcoin::Network, TxHeight};
use bdk_esplora::esplora_client::{self, BlockingClient};
use bdk_esplora::EsploraClientExt;

use std::{
    io::{self, Write},
    num::NonZeroU8,
};

use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Parser, Subcommand},
};

#[derive(Subcommand, Debug, Clone)]
enum EsploraCommands {
    /// Scans the addresses in the wallet using esplora API.
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: usize,

        #[clap(flatten)]
        scan_options: ScanOptions,
    },
    /// Scans particular addresses using esplora API
    Sync {
        /// Scan all the unused addresses
        #[clap(long)]
        unused_spks: bool,
        /// Scan the script addresses that have unspent outputs
        #[clap(long)]
        unspent_spks: bool,
        /// Scan every address that you have derived
        #[clap(long)]
        all_spks: bool,
        /// Scan unspent outpoints for spends or changes to confirmation status of residing tx
        #[clap(long)]
        unspent_outpoints: bool,
        /// Scan unconfirmed transactions for updates
        #[clap(long)]
        unconfirmed_txs: bool,

        #[clap(flatten)]
        scan_options: ScanOptions,
    },
}

#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ScanOptions {
    #[clap(long, default_value = "5")]
    pub parallel_requests: NonZeroU8,
}

struct WrappedClient(BlockingClient);

impl std::ops::Deref for WrappedClient {
    type Target = BlockingClient;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl bdk_cli::Broadcast for WrappedClient {
    type Error = bdk_esplora::esplora_client::Error;

    fn broadcast(&self, tx: &bdk_chain::bitcoin::Transaction) -> anyhow::Result<(), Self::Error> {
        self.0.broadcast(tx)
    }
}

impl WrappedClient {
    pub fn new(base_url: &str) -> Result<Self, esplora_client::Error> {
        esplora_client::Builder::new(base_url)
            .build_blocking()
            .map(Self)
    }
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, keychain_tracker, db) = bdk_cli::init::<EsploraCommands, _>()?;
    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3002",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let client = WrappedClient::new(esplora_url)?;

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
        EsploraCommands::Scan {
            stop_gap,
            scan_options,
        } => {
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
                .scan(
                    &local_chain,
                    spk_iterators,
                    core::iter::empty(),
                    core::iter::empty(),
                    Some(stop_gap),
                    scan_options.parallel_requests,
                )
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
            mut unused_spks,
            mut unspent_spks,
            all_spks,
            unspent_outpoints,
            unconfirmed_txs,
            scan_options,
        } => {
            // Get a short lock on the tracker to get the spks we're interested in
            let tracker = keychain_tracker.lock().unwrap();

            if !(all_spks || unused_spks || unspent_spks || unspent_outpoints || unconfirmed_txs) {
                unused_spks = true;
                unspent_spks = true;
            } else if all_spks {
                unused_spks = false;
                unspent_spks = false
            }

            let mut spks: Box<dyn Iterator<Item = bdk_chain::bitcoin::Script>> =
                Box::new(core::iter::empty());
            if all_spks {
                let all_spks = tracker
                    .txout_index
                    .all_spks()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>();
                spks = Box::new(spks.chain(all_spks.into_iter().map(|(index, script)| {
                    eprintln!("scanning {:?}", index);
                    script
                })));
            }
            if unused_spks {
                let unused_spks = tracker
                    .txout_index
                    .unused_spks(..)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>();
                spks = Box::new(spks.chain(unused_spks.into_iter().map(|(index, script)| {
                    eprintln!("Checking if address at {:?} has been used", index);
                    script
                })));
            }
            if unspent_spks {
                let unspent_txouts = tracker
                    .full_utxos()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<Vec<_>>();
                spks = Box::new(
                    spks.chain(unspent_txouts.into_iter().map(|(_index, ftxout)| {
                        eprintln!("checking if {} has been spent", ftxout.outpoint);
                        ftxout.txout.script_pubkey
                    })),
                );
            }

            let outpoints: Vec<_> = match unspent_outpoints {
                true => tracker.full_txouts().map(|(_, txo)| txo.outpoint).collect(),
                false => Default::default(),
            };

            let txids: Vec<_> = match unconfirmed_txs {
                true => tracker
                    .chain()
                    .range_txids_by_height(TxHeight::Unconfirmed..)
                    .map(|(_, txid)| *txid)
                    .collect(),
                false => Default::default(),
            };

            let local_chain = tracker.chain().checkpoints().clone();

            // drop lock on tracker
            drop(tracker);

            // we scan the desired spks **without** a lock on the tracker
            let scan = client
                .scan_without_keychain(
                    &local_chain,
                    spks,
                    txids,
                    outpoints,
                    scan_options.parallel_requests,
                )
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
