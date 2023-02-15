pub mod esplora;

use std::collections::HashMap;
use std::io::{self, Write};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use bdk_chain::bitcoin::{secp256k1::Secp256k1, Address, Network};
use bdk_chain::chain_graph::ChainGraph;
use bdk_chain::file_store::KeychainStore;
use bdk_chain::keychain::KeychainScan;
use bdk_chain::keychain::{KeychainChangeSet, KeychainTracker};
use bdk_chain::miniscript::descriptor::DescriptorSecretKey;
use bdk_chain::miniscript::{Descriptor, DescriptorPublicKey};
use bdk_chain::sparse_chain::ChainPosition;
use bdk_chain::ConfirmationTime;
pub use bdk_chain::*;
use bdk_cli::anyhow::{Context, Result};
use bdk_cli::Keychain;
use esplora::Client;

pub const DEFAULT_PARALLEL_REQUESTS: u8 = 5;

pub fn init() -> Result<(
    HashMap<DescriptorPublicKey, DescriptorSecretKey>,
    Mutex<KeychainTracker<Keychain, ConfirmationTime>>,
    Mutex<KeychainStore<Keychain, ConfirmationTime>>,
)>
where
    KeychainChangeSet<Keychain, ConfirmationTime>: serde::Serialize + serde::de::DeserializeOwned,
{
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)")?;

    let mut tracker = KeychainTracker::default();

    tracker
        .txout_index
        .add_keychain(Keychain::External, descriptor);

    let (internal_descriptor, internal_keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)"
    )?;

    keymap.extend(internal_keymap);
    tracker
        .txout_index
        .add_keychain(Keychain::Internal, internal_descriptor);

    tracker.txout_index.set_lookahead_for_all(10);
    tracker.set_checkpoint_limit(Some(50));

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut dir = std::env::temp_dir();
    dir.push(format!("db_{}", time.as_nanos()));

    let mut db = KeychainStore::<Keychain, ConfirmationTime>::new_from_path(dir.to_str().unwrap())?;

    if let Err(e) = db.load_into_keychain_tracker(&mut tracker) {
        match tracker.chain().latest_checkpoint()  {
            Some(checkpoint) => eprintln!("Failed to load all changesets from {}. Last checkpoint was at height {}. Error: {}", dir.to_str().unwrap(), checkpoint.height, e),
            None => eprintln!("Failed to load any checkpoints from {}: {}", dir.to_str().unwrap(), e),

        }
        eprintln!("⚠ Consider running a rescan of chain data.");
    }

    Ok((keymap, Mutex::new(tracker), Mutex::new(db)))
}

pub fn get_new_address(
    tracker: &Mutex<KeychainTracker<Keychain, ConfirmationTime>>,
    db: &Mutex<KeychainStore<Keychain, ConfirmationTime>>,
    network: Network,
) -> Result<Address>
where
    KeychainChangeSet<Keychain, ConfirmationTime>: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut tracker = tracker.lock().unwrap();
    let txout_index = &mut tracker.txout_index;

    let ((index, spk), additions) = txout_index.next_unused_spk(&Keychain::External);

    let mut db = db.lock().unwrap();
    // update database since we're about to give out a new address
    db.append_changeset(&additions.into())?;
    let spk = spk.clone();
    let address =
        Address::from_script(&spk, network).expect("should always be able to derive address");
    eprintln!("This is the address at index {}", index);
    println!("{}", address);
    Ok(address)
}

pub fn get_balance(tracker: &Mutex<KeychainTracker<Keychain, ConfirmationTime>>) -> (u64, u64) {
    let tracker = tracker.lock().unwrap();
    let (confirmed, unconfirmed) =
        tracker
            .full_utxos()
            .fold((0, 0), |(confirmed, unconfirmed), (_, utxo)| {
                if utxo.chain_position.height().is_confirmed() {
                    (confirmed + utxo.txout.value, unconfirmed)
                } else {
                    (confirmed, unconfirmed + utxo.txout.value)
                }
            });

    (confirmed, unconfirmed)
}

pub fn run_sync<P>(
    mut unused: bool,
    mut unspent: bool,
    all: bool,
    keychain_tracker: &Mutex<KeychainTracker<Keychain, P>>,
    db: &Mutex<KeychainStore<Keychain, P>>,
    client: &Client,
) -> Result<()>
where
    P: ChainPosition,
    KeychainChangeSet<Keychain, P>: serde::Serialize + serde::de::DeserializeOwned,
    KeychainScan<Keychain, P>: From<ChainGraph<ConfirmationTime>>,
{
    let (spks, local_chain) = {
        // Get a short lock on the tracker to get the spks we're interested in
        let tracker = keychain_tracker.lock().unwrap();
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
            spks = Box::new(
                spks.chain(unspent_txouts.into_iter().map(|(_index, ftxout)| {
                    eprintln!("checking if {} has been spent", ftxout.outpoint);
                    ftxout.txout.script_pubkey
                })),
            );
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

    Ok(())
}

pub fn run_scan<P>(
    stop_gap: usize,
    keychain_tracker: &Mutex<KeychainTracker<Keychain, P>>,
    db: &Mutex<KeychainStore<Keychain, P>>,
    client: &Client,
) -> Result<()>
where
    P: ChainPosition,
    KeychainChangeSet<Keychain, P>: serde::Serialize + serde::de::DeserializeOwned,
    KeychainScan<Keychain, P>: From<KeychainScan<Keychain, ConfirmationTime>>,
{
    let (spk_iterators, local_chain) = {
        // Get a short lock on the tracker to get the spks iterators
        // and local chain state
        let tracker = keychain_tracker.lock().unwrap();
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
        let changeset = tracker.apply_update(wallet_scan.into())?;
        db.append_changeset(&changeset)?;
    }

    Ok(())
}
