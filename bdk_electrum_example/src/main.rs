mod electrum;
use bdk_chain::{
    bitcoin::Network, keychain::KeychainChangeSet, sparse_chain::SparseChain, BlockPosition,
    TxHeight,
};
use electrum::ElectrumClient;

use bdk_cli::{
    anyhow::{self, anyhow, Context},
    clap::{self, Parser, Subcommand},
    Keychain,
};
use std::{fmt::Debug, io, io::Write};

use electrum_client::{Client, ConfigBuilder, ElectrumApi};

#[derive(Subcommand, Debug, Clone)]
enum ElectrumCommands {
    /// Scans the addresses in the wallet using electrum API.
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: usize,
        #[clap(flatten)]
        scan_option: ScanOption,
    },
    /// Scans the addresses in the wallet using electrum API.
    /// Stores it in [`BlockPosition`] index.
    ScanBlockPosition {
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
    let (args, keymap, mut tracker, mut db) = bdk_cli::init::<ElectrumCommands, _>()?;

    let (tracker_bp, db_bp) =
        if let bdk_cli::Commands::ChainSpecific(ElectrumCommands::ScanBlockPosition { .. }) =
            &args.command
        {
            let (_, _, tracker_bp, db_bp) = bdk_cli::init::<ElectrumCommands, BlockPosition>()?;
            (Some(tracker_bp), Some(db_bp))
        } else {
            (None, None)
        };

    let electrum_url = match args.network {
        Network::Bitcoin => "ssl://electrum.blockstream.info:50002",
        Network::Testnet => "ssl://electrum.blockstream.info:60002",
        Network::Regtest => "ssl://localhost:60401",
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

    let mut keychain_changeset = KeychainChangeSet::default();

    let chain_update = match electrum_cmd {
        ElectrumCommands::Scan {
            stop_gap,
            scan_option,
        } => {
            let scripts = tracker
                .txout_index
                .scripts_of_all_keychains()
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

            let (new_sparsechain, keychain_index_update) = client.wallet_txid_scan(
                scripts,
                Some(stop_gap),
                tracker.chain().checkpoints(),
                scan_option.batch_size,
            )?;

            eprintln!();

            keychain_changeset.derivation_indices = keychain_index_update;

            new_sparsechain
        }
        ElectrumCommands::Sync {
            mut unused,
            mut unspent,
            all,
            scan_option,
        } => {
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
            if unused {
                spks = Box::new(spks.chain(txout_index.inner().unused(..).map(
                    |(index, script)| {
                        eprintln!("Checking if address at {:?} has been used", index);
                        script.clone()
                    },
                )));
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
                spks = Box::new(spks.chain(tracker.full_utxos().map(|(_index, ftxout)| {
                    eprintln!("checking if {} has been spent", ftxout.outpoint);
                    ftxout.txout.script_pubkey
                })));
            }

            let new_sparsechain = client
                .spk_txid_scan(spks, tracker.chain().checkpoints(), scan_option.batch_size)
                .context("scanning the blockchain")?;

            new_sparsechain
        }
        ElectrumCommands::ScanBlockPosition {
            stop_gap,
            scan_option,
        } => {
            if let (Some(mut tracker_bp), Some(mut db_bp)) = (tracker_bp, db_bp) {
                let scripts = tracker_bp.txout_index.script_pubkeys_of_all_keychains();
                let (new_sparsechain, keychain_index_update) = client
                    .wallet_scan_block_position::<Keychain, BlockPosition>(
                        scripts,
                        Some(stop_gap),
                        tracker_bp.chain().checkpoints(),
                        scan_option.batch_size,
                    )?;

                let mut kp = KeychainChangeSet::<Keychain, BlockPosition>::default();

                kp.derivation_indices = keychain_index_update;

                let sparsechain_changeset =
                    match tracker_bp.chain().determine_changeset(&new_sparsechain) {
                        Ok(cs) => cs,
                        Err(update_failure) => {
                            return Err(anyhow!(
                                "Changeset determination failed : {:?}",
                                update_failure
                            ))
                        }
                    };

                let new_txids = tracker_bp
                    .chain()
                    .changeset_additions(&sparsechain_changeset)
                    .collect::<Vec<_>>();

                let new_txs = client.batch_transaction_get(new_txids.iter())?;

                let chaingraph_changeset = match tracker_bp
                    .chain_graph()
                    .inflate_changeset(sparsechain_changeset, new_txs)
                {
                    Ok(cs) => cs,
                    Err(cs_error) => {
                        return Err(anyhow!(
                            "Chaingraph changeset creation failed : {:?}",
                            cs_error
                        ))
                    }
                };

                kp.chain_graph = chaingraph_changeset;

                db_bp.append_changeset(&kp)?;
                if let Err(err) = tracker_bp.apply_changeset(kp) {
                    return Err(anyhow!(
                        "Changeset application failed in tracker_bp : {:?}",
                        err
                    ));
                }
                println!("sync completed!!");
            }

            SparseChain::<TxHeight>::default()
        }
    };

    let sparsechain_changeset = tracker.chain().determine_changeset(&chain_update)?;

    let new_txids = tracker
        .chain()
        .changeset_additions(&sparsechain_changeset)
        .collect::<Vec<_>>();

    let new_txs = client
        .batch_transaction_get(new_txids.iter())
        .context("fetching full transactions")?;

    let chaingraph_changeset = tracker
        .chain_graph()
        .inflate_changeset(sparsechain_changeset, new_txs)
        .context("inflating changeset")?;

    keychain_changeset.chain_graph = chaingraph_changeset;

    db.append_changeset(&keychain_changeset)?;
    tracker.apply_changeset(keychain_changeset);
    Ok(())
}
