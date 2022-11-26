mod esplora;
use crate::esplora::Client;
use bdk_core::{
    bitcoin::{secp256k1::Secp256k1, Network},
    ConfirmationTime,
};
use bdk_file_store::KeychainStore;
use bdk_keychain::{
    miniscript::{Descriptor, DescriptorPublicKey},
    KeychainTracker,
};

use std::io::{self, Write};

const DEFAULT_PARALLEL_REQUESTS: u8 = 5;

use bdk_cli::{handle_commands, Args, Commands, Context, Keychain, Parser, Result};

fn main() -> Result<()> {
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;

    let mut keychain_tracker = KeychainTracker::default();
    keychain_tracker
        .txout_index
        .add_keychain(Keychain::External, descriptor);

    let internal = args
        .change_descriptor
        .map(|descriptor| Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &descriptor))
        .transpose()?;

    let change_keychain = if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        keychain_tracker
            .txout_index
            .add_keychain(Keychain::Internal, internal_descriptor);
        Keychain::Internal
    } else {
        Keychain::External
    };

    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3000",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let mut db = KeychainStore::<Keychain, ConfirmationTime>::load(
        args.db_dir.as_path(),
        &mut keychain_tracker,
    )?;
    let client = Client::new(esplora_url, DEFAULT_PARALLEL_REQUESTS)?;

    match args.command {
        Commands::Scan => {
            let stop_gap = 10;

            let spk_iterators = keychain_tracker
                .txout_index
                .iter_all_spks()
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

            let wallet_scan = client
                .wallet_scan(
                    spk_iterators,
                    Some(stop_gap),
                    keychain_tracker.chain().checkpoints().clone(),
                )
                .context("scanning the blockchain")?;
            eprintln!();

            let changeset = keychain_tracker.determine_changeset(&wallet_scan)?;
            db.append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
        }
        Commands::Sync {
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
                spks = Box::new(spks.chain(keychain_tracker.utxos().map(|(_index, ftxout)| {
                    eprintln!("checking if {} has been spent", ftxout.outpoint);
                    ftxout.txout.script_pubkey
                })));
            }

            let scan = client
                .spk_scan(spks, keychain_tracker.chain().checkpoints().clone())
                .context("scanning the blockchain")?;

            let changeset = keychain_tracker
                .chain_graph()
                .determine_changeset(&scan)?
                .into();
            db.append_changeset(&changeset)?;
            keychain_tracker.apply_changeset(changeset);
        }
        // For everything else run handler
        _ => handle_commands(
            args.command,
            client,
            &mut keychain_tracker,
            &mut db,
            args.network,
            &keymap,
            &Keychain::External,
            &change_keychain,
        )?,
    }
    Ok(())
}
