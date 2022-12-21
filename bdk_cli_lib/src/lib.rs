pub extern crate anyhow;
use anyhow::{anyhow, Result};
use bdk_core::{
    bitcoin::{
        secp256k1::Secp256k1,
        util::sighash::{Prevouts, SighashCache},
        Address, LockTime, Network, Sequence, Transaction, TxIn, TxOut,
    },
    coin_select::{coin_select_bnb, CoinSelector, CoinSelectorOpt, WeightedValue},
    sparse_chain::{self, ChainIndex},
};
use bdk_file_store::KeychainStore;
use bdk_keychain::{
    miniscript::{
        descriptor::{DescriptorSecretKey, KeyMap},
        Descriptor, DescriptorPublicKey,
    },
    DescriptorExt, KeychainChangeSet, KeychainTracker,
};
pub use clap;
use clap::{Parser, Subcommand};
use std::{cmp::Reverse, collections::HashMap, fmt::Debug, path::PathBuf, time::Duration};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
pub struct Args<A: clap::Args, C: clap::Subcommand> {
    #[clap(env = "DESCRIPTOR")]
    pub descriptor: String,
    #[clap(env = "CHANGE_DESCRIPTOR")]
    pub change_descriptor: Option<String>,

    #[clap(env = "BITCOIN_NETWORK", long, default_value = "signet")]
    pub network: Network,

    #[clap(env = "BDK_DB_DIR", long, default_value = ".bdk_example_db")]
    pub db_dir: PathBuf,

    #[clap(env = "BDK_CP_LIMIT", long, default_value = "20")]
    pub cp_limit: usize,

    #[clap(flatten)]
    pub chain_args: A,

    #[clap(subcommand)]
    pub command: Commands<C>,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Commands<C: clap::Subcommand> {
    #[clap(flatten)]
    ChainSpecific(C),
    /// Address generation and inspection
    Address {
        #[clap(subcommand)]
        addr_cmd: AddressCmd,
    },
    /// Get the wallet balance
    Balance,
    /// TxOut related commands
    #[clap(name = "txout")]
    TxOut {
        #[clap(subcommand)]
        txout_cmd: TxOutCmd,
    },
    /// Send coins to an address
    Send {
        value: u64,
        address: Address,
        #[clap(short, default_value = "largest-first")]
        coin_select: CoinSelectionAlgo,
    },
}

#[derive(Clone, Debug)]
pub enum CoinSelectionAlgo {
    LargestFirst,
    SmallestFirst,
    OldestFirst,
    NewestFirst,
    BranchAndBound,
}

impl Default for CoinSelectionAlgo {
    fn default() -> Self {
        Self::LargestFirst
    }
}

impl core::str::FromStr for CoinSelectionAlgo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use CoinSelectionAlgo::*;
        Ok(match s {
            "largest-first" => LargestFirst,
            "smallest-first" => SmallestFirst,
            "oldest-first" => OldestFirst,
            "newest-first" => NewestFirst,
            "bnb" => BranchAndBound,
            unknown => return Err(anyhow!("unknown coin selection algorithm '{}'", unknown)),
        })
    }
}

impl core::fmt::Display for CoinSelectionAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use CoinSelectionAlgo::*;
        write!(
            f,
            "{}",
            match self {
                LargestFirst => "largest-first",
                SmallestFirst => "smallest-first",
                OldestFirst => "oldest-first",
                NewestFirst => "newest-first",
                BranchAndBound => "bnb",
            }
        )
    }
}

#[derive(Subcommand, Debug, Clone)]
pub enum AddressCmd {
    /// Get the next unused address
    Next,
    /// Get a new address regardless if the existing ones haven't been used
    New,
    /// List all addresses
    List {
        #[clap(long)]
        change: bool,
    },
    Index,
}

#[derive(Subcommand, Debug, Clone)]
pub enum TxOutCmd {
    List,
}

#[derive(
    Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, serde::Deserialize, serde::Serialize,
)]
pub enum Keychain {
    External,
    Internal,
}

impl core::fmt::Display for Keychain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Keychain::External => write!(f, "external"),
            Keychain::Internal => write!(f, "internal"),
        }
    }
}

/// A structure defining output of a AddressCmd execution.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct AddrsOutput {
    keychain: String,
    index: u32,
    addrs: Address,
    used: bool,
}

pub fn run_address_cmd<I>(
    keychain_tracker: &mut KeychainTracker<Keychain, I>,
    db: &mut KeychainStore<Keychain, I>,
    addr_cmd: AddressCmd,
    network: Network,
) -> Result<()>
where
    I: sparse_chain::ChainIndex,
    KeychainChangeSet<Keychain, I>: serde::Serialize + serde::de::DeserializeOwned,
{
    let txout_index = &mut keychain_tracker.txout_index;

    let new_address = match addr_cmd {
        AddressCmd::Next => Some(txout_index.derive_next_unused(&Keychain::External)),
        AddressCmd::New => Some(txout_index.derive_new(&Keychain::External)),
        _ => None,
    };

    if let Some((index, spk)) = new_address {
        let spk = spk.clone();
        // update database since we're about to give out a new address
        db.set_derivation_indices(txout_index.derivation_indices())?;
        let address =
            Address::from_script(&spk, network).expect("should always be able to derive address");
        eprintln!("This is the address at index {}", index);
        println!("{}", address);
    }

    match addr_cmd {
        AddressCmd::Next | AddressCmd::New => {
            /* covered */
            Ok(())
        }
        AddressCmd::Index => {
            for (keychain, derivation_index) in txout_index.derivation_indices() {
                println!("{:?}: {}", keychain, derivation_index);
            }
            Ok(())
        }
        AddressCmd::List { change } => {
            let target_keychain = match change {
                true => Keychain::Internal,
                false => Keychain::External,
            };
            for (index, spk) in txout_index.script_pubkeys_by_keychain(&target_keychain) {
                let address = Address::from_script(&spk, network)
                    .expect("should always be able to derive address");
                println!(
                    "{} used:{}",
                    address,
                    txout_index.is_used(&(target_keychain, index))
                );
            }
            Ok(())
        }
    }
}

pub fn run_balance_cmd<I: ChainIndex>(keychain_tracker: &KeychainTracker<Keychain, I>) {
    let (confirmed, unconfirmed) =
        keychain_tracker
            .full_utxos()
            .fold((0, 0), |(confirmed, unconfirmed), (_, utxo)| {
                if utxo.chain_index.height().is_confirmed() {
                    (confirmed + utxo.txout.value, unconfirmed)
                } else {
                    (confirmed, unconfirmed + utxo.txout.value)
                }
            });

    println!("confirmed: {}", confirmed);
    println!("unconfirmed: {}", unconfirmed);
}

pub fn run_txo_cmd<K: Debug + Clone + Ord, I: ChainIndex>(
    txout_cmd: TxOutCmd,
    keychain_tracker: &KeychainTracker<K, I>,
    network: Network,
) {
    match txout_cmd {
        TxOutCmd::List => {
            for (spk_index, full_txout) in keychain_tracker.full_txouts() {
                let address =
                    Address::from_script(&full_txout.txout.script_pubkey, network).unwrap();

                println!(
                    "{:?} {} {} {} spent:{:?}",
                    spk_index,
                    full_txout.txout.value,
                    full_txout.outpoint,
                    address,
                    full_txout.spent_by
                )
            }
        }
    }
}

pub fn create_tx<I: ChainIndex>(
    value: u64,
    address: Address,
    coin_select: CoinSelectionAlgo,
    keychain_tracker: &mut KeychainTracker<Keychain, I>,
    keymap: &HashMap<DescriptorPublicKey, DescriptorSecretKey>,
) -> Result<Transaction> {
    use bdk_keychain::miniscript::plan::*;
    let assets = Assets {
        keys: keymap.iter().map(|(pk, _)| pk.clone()).collect(),
        ..Default::default()
    };

    let mut candidates = keychain_tracker.planned_utxos(&assets).collect::<Vec<_>>();

    // apply coin selection algorithm
    match coin_select {
        CoinSelectionAlgo::LargestFirst => {
            candidates.sort_by_key(|(_, utxo)| Reverse(utxo.txout.value))
        }
        CoinSelectionAlgo::SmallestFirst => candidates.sort_by_key(|(_, utxo)| utxo.txout.value),
        CoinSelectionAlgo::OldestFirst => {
            candidates.sort_by_key(|(_, utxo)| utxo.chain_index.clone())
        }
        CoinSelectionAlgo::NewestFirst => {
            candidates.sort_by_key(|(_, utxo)| Reverse(utxo.chain_index.clone()))
        }
        CoinSelectionAlgo::BranchAndBound => {}
    }

    // turn the txos we chose into a weight and value
    let wv_candidates = candidates
        .iter()
        .map(|(plan, utxo)| {
            WeightedValue::new(
                utxo.txout.value,
                plan.expected_weight() as _,
                plan.witness_version().is_some(),
            )
        })
        .collect();

    let mut outputs = vec![TxOut {
        value,
        script_pubkey: address.script_pubkey(),
    }];

    let internal_keychain = if keychain_tracker
        .txout_index
        .keychains()
        .get(&Keychain::Internal)
        .is_some()
    {
        Keychain::Internal
    } else {
        Keychain::External
    };

    let (change_index, change_script) = {
        let (index, script) = keychain_tracker
            .txout_index
            .derive_next_unused(&internal_keychain);
        (index, script.clone())
    };
    let change_plan = keychain_tracker
        .txout_index
        .keychains()
        .get(&internal_keychain)
        .expect("must exist")
        .at_derivation_index(change_index)
        .plan_satisfaction(&assets)
        .expect("failed to obtain change plan");

    let mut change_output = TxOut {
        value: 0,
        script_pubkey: change_script,
    };

    let cs_opts = CoinSelectorOpt {
        target_feerate: 0.5,
        min_drain_value: keychain_tracker
            .txout_index
            .keychains()
            .get(&internal_keychain)
            .expect("must exist")
            .dust_value(),
        ..CoinSelectorOpt::fund_outputs(
            &outputs,
            &change_output,
            change_plan.expected_weight() as u32,
        )
    };

    // TODO: How can we make it easy to shuffle in order of inputs and outputs here?
    // apply coin selection by saying we need to fund these outputs
    let mut coin_selector = CoinSelector::new(&wv_candidates, &cs_opts);

    // just select coins in the order provided until we have enough
    // only use first result (least waste)
    let selection = match coin_select {
        CoinSelectionAlgo::BranchAndBound => {
            coin_select_bnb(Duration::from_secs(10), coin_selector.clone())
                .map_or_else(|| coin_selector.select_until_finished(), |cs| cs.finish())?
        }
        _ => coin_selector.select_until_finished()?,
    };
    let (_, selection_meta) = selection.best_strategy();

    // get the selected utxos
    let selected_txos = selection.apply_selection(&candidates).collect::<Vec<_>>();

    if let Some(drain_value) = selection_meta.drain_value {
        change_output.value = drain_value;
        // if the selection tells us to use change and the change value is sufficient we add it as an output
        outputs.push(change_output)
    }

    let mut transaction = Transaction {
        version: 0x02,
        lock_time: keychain_tracker
            .chain()
            .latest_checkpoint()
            .and_then(|block_id| LockTime::from_height(block_id.height).ok())
            .unwrap_or(LockTime::ZERO)
            .into(),
        input: selected_txos
            .iter()
            .map(|(_, utxo)| TxIn {
                previous_output: utxo.outpoint,
                sequence: Sequence::ENABLE_RBF_NO_LOCKTIME,
                ..Default::default()
            })
            .collect(),
        output: outputs,
    };

    let prevouts = selected_txos
        .iter()
        .map(|(_, utxo)| utxo.txout.clone())
        .collect::<Vec<_>>();
    let sighash_prevouts = Prevouts::All(&prevouts);

    // first set tx values for plan so that we don't change them while signing
    for (i, (plan, _)) in selected_txos.iter().enumerate() {
        if let Some(sequence) = plan.required_sequence() {
            transaction.input[i].sequence = sequence
        }
    }

    // create a short lived transaction
    let _sighash_tx = transaction.clone();
    let mut sighash_cache = SighashCache::new(&_sighash_tx);

    for (i, (plan, _)) in selected_txos.iter().enumerate() {
        let requirements = plan.requirements();
        let mut auth_data = SatisfactionMaterial::default();
        assert!(
            !requirements.requires_hash_preimages(),
            "can't have hash pre-images since we didn't provide any"
        );
        assert!(
            requirements.signatures.sign_with_keymap(
                i,
                &keymap,
                &sighash_prevouts,
                None,
                None,
                &mut sighash_cache,
                &mut auth_data,
                &Secp256k1::default(),
            )?,
            "we should have signed with this input"
        );

        match plan.try_complete(&auth_data) {
            PlanState::Complete {
                final_script_sig,
                final_script_witness,
            } => {
                if let Some(witness) = final_script_witness {
                    transaction.input[i].witness = witness;
                }

                if let Some(script_sig) = final_script_sig {
                    transaction.input[i].script_sig = script_sig;
                }
            }
            PlanState::Incomplete(_) => {
                return Err(anyhow!(
                    "we weren't able to complete the plan with our keys"
                ));
            }
        }
    }

    Ok(transaction)
}

pub trait Broadcast {
    type Error: std::error::Error + Send + Sync + 'static;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error>;
}

pub fn handle_commands<C: clap::Subcommand, I>(
    command: Commands<C>,
    client: impl Broadcast,
    tracker: &mut KeychainTracker<Keychain, I>,
    store: &mut KeychainStore<Keychain, I>,
    network: Network,
    keymap: &HashMap<DescriptorPublicKey, DescriptorSecretKey>,
) -> Result<()>
where
    I: ChainIndex,
    KeychainChangeSet<Keychain, I>: serde::Serialize + serde::de::DeserializeOwned,
{
    match command {
        // TODO: Make these functions return stuffs
        Commands::Address { addr_cmd } => {
            run_address_cmd(tracker, store, addr_cmd, network)?;
        }
        Commands::Balance => {
            run_balance_cmd(&tracker);
        }
        Commands::TxOut { txout_cmd } => {
            run_txo_cmd(txout_cmd, tracker, network);
        }
        Commands::Send {
            value,
            address,
            coin_select,
        } => {
            let transaction = create_tx(value, address, coin_select, tracker, &keymap)?;
            client.broadcast(&transaction)?;
            println!("Broadcasted Tx : {}", transaction.txid());
        }
        Commands::ChainSpecific(_) => {
            todo!("example code is meant to handle this!")
        }
    }

    Ok(())
}

pub fn init<A: clap::Args, C: clap::Subcommand, I>() -> anyhow::Result<(
    Args<A, C>,
    KeyMap,
    KeychainTracker<Keychain, I>,
    KeychainStore<Keychain, I>,
)>
where
    I: sparse_chain::ChainIndex,
    KeychainChangeSet<Keychain, I>: serde::Serialize + serde::de::DeserializeOwned,
{
    let args = Args::<A, C>::parse();
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

    if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        keychain_tracker
            .txout_index
            .add_keychain(Keychain::Internal, internal_descriptor);
    };

    let db = KeychainStore::<Keychain, I>::load(args.db_dir.as_path(), &mut keychain_tracker)?;

    Ok((Args::parse(), keymap, keychain_tracker, db))
}
