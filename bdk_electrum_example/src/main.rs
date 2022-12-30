mod electrum;

use std::fmt::Debug;

use bdk_chain::{bitcoin::Network, keychain::KeychainChangeSet};
use electrum::ElectrumClient;

use bdk_cli::{
    anyhow::{self, anyhow, Context},
    clap::{self, Parser, Subcommand},
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

    let mut keychain_changeset = KeychainChangeSet::default();

    let chain_update = match electrum_cmd {
        ElectrumCommands::Scan {
            stop_gap,
            scan_option,
        } => {
            let scripts = tracker.txout_index.iter_all_script_pubkeys_by_keychain();

            let (new_sparsechain, keychain_index_update) = client.wallet_txid_scan(
                scripts,
                Some(stop_gap),
                tracker.chain().checkpoints(),
                scan_option.batch_size,
            )?;

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
    };

    let sparsechain_changeset = match tracker.chain().determine_changeset(&chain_update) {
        Ok(cs) => cs,
        Err(update_failure) => {
            return Err(anyhow!(
                "Changeset determination failed : {:?}",
                update_failure
            ))
        }
    };

    let new_txids = tracker
        .chain()
        .changeset_additions(&sparsechain_changeset)
        .collect::<Vec<_>>();

    let new_txs = client.batch_transaction_get(new_txids.iter())?;

    let chaingraph_changeset = match tracker
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

    keychain_changeset.chain_graph = chaingraph_changeset;

    db.append_changeset(&keychain_changeset)?;
    if let Err(err) = tracker.apply_changeset(keychain_changeset) {
        return Err(anyhow!(
            "Changeset application failed in tracker : {:?}",
            err
        ));
    }
    debug!("sync completed!!");
    Ok(())
}
