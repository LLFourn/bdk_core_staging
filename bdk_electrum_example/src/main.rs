use bdk_chain::{bitcoin::Network, TxHeight};
use bdk_cli::{
    anyhow::{self, Context},
    clap::{self, Parser, Subcommand},
    Broadcast,
};
use bdk_electrum::{
    electrum_client::{self, ElectrumApi},
    ScanParams, ScanParamsWithoutKeychain,
};
use std::{collections::BTreeMap, fmt::Debug, io, io::Write, ops::Deref};

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
        scan_option: ScanOption,
    },
}

#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ScanOption {
    /// Set batch size for each script_history call to electrum client
    #[clap(long, default_value = "25")]
    pub batch_size: usize,
}

/// A wrapped [`bdk_electrum::ElectrumClient`] that implements [`bdk_cli::Broadcast`].
struct WrappedClient(bdk_electrum::ElectrumClient<TxHeight>);

impl Deref for WrappedClient {
    type Target = bdk_electrum::ElectrumClient<TxHeight>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl WrappedClient {
    fn new(url: &str, config: electrum_client::Config) -> Result<Self, bdk_electrum::Error> {
        bdk_electrum::ElectrumClient::from_config(url, config).map(Self)
    }
}

impl Broadcast for WrappedClient {
    type Error = bdk_electrum::Error;

    fn broadcast(&self, tx: &bdk_chain::bitcoin::Transaction) -> anyhow::Result<(), Self::Error> {
        self.transaction_broadcast(tx)?;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<ElectrumCommands, _>()?;

    let electrum_url = match args.network {
        Network::Bitcoin => "ssl://electrum.blockstream.info:50002",
        Network::Testnet => "ssl://electrum.blockstream.info:60002",
        Network::Regtest => "tcp://localhost:60401",
        Network::Signet => "tcp://signet-electrumx.wakiyamap.dev:50001",
    };
    let config = electrum_client::Config::builder()
        .validate_domain(match args.network {
            Network::Bitcoin => true,
            _ => false,
        })
        .build();

    let client = WrappedClient::new(electrum_url, config)?;

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

    let response = match electrum_cmd {
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
                    .collect::<BTreeMap<_, _>>();
                let local_chain = tracker.chain().checkpoints().clone();
                (spk_iterators, local_chain)
            };

            let params = ScanParams {
                keychain_spks: spk_iterators,
                stop_gap,
                batch_size: scan_option.batch_size,
                ..Default::default()
            };

            // we scan the spks **without** a lock on the tracker
            client.scan(&local_chain, params)?
        }
        ElectrumCommands::Sync {
            mut unused_spks,
            mut unspent_spks,
            all_spks,
            unspent_outpoints,
            unconfirmed_txs,
            scan_option,
        } => {
            // Get a short lock on the tracker to get the spks we're interested in
            let tracker = tracker.lock().unwrap();

            if !(all_spks || unused_spks || unspent_spks || unspent_outpoints || unconfirmed_txs) {
                unused_spks = true;
                unspent_spks = true;
            } else if all_spks {
                unused_spks = false;
                unspent_spks = false;
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

            let mut outpoints = Vec::new();
            if unspent_outpoints {
                outpoints = tracker.full_utxos().map(|(_, txo)| txo.outpoint).collect();
            }

            let mut txs = Vec::new();
            if unconfirmed_txs {
                txs = tracker
                    .chain()
                    .range_txids_by_height(TxHeight::Unconfirmed..)
                    .filter_map(|(_, txid)| tracker.graph().get_tx(*txid))
                    .cloned()
                    .collect();
            }

            let local_chain = tracker.chain().checkpoints().clone();

            // drop lock on tracker
            drop(tracker);

            // we scan the spks **without** a lock on the tracker
            let params = ScanParamsWithoutKeychain {
                batch_size: scan_option.batch_size,
                txs,
                outpoints,
                ..ScanParamsWithoutKeychain::from_spks(spks)
            };
            client
                .scan_without_keychain(&local_chain, params)
                .context("scanning the blockchain")?
                .into_with_keychain()
        }
    };

    let missing_txids = response.find_missing_txids(&*tracker.lock().unwrap());

    // fetch the missing full transactions **without** a lock on the tracker
    let new_txs = client
        .batch_transaction_get(missing_txids)
        .context("fetching full transactions")?;

    {
        // Get a final short lock to apply the changes
        let mut tracker = tracker.lock().unwrap();
        let changeset = response.into_keychain_changeset(new_txs, &*tracker)?;
        db.lock().unwrap().append_changeset(&changeset)?;
        tracker.apply_changeset(changeset);
    };

    Ok(())
}
