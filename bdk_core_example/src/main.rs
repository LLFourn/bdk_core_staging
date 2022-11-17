use anyhow::{anyhow, Context};
use bdk_core::{
    bitcoin::{
        consensus,
        hashes::hex::ToHex,
        secp256k1::Secp256k1,
        util::sighash::{Prevouts, SighashCache},
        Address, LockTime, Network, Sequence, Transaction, TxIn, TxOut,
    },
    chain_graph::ChainGraph,
    coin_select::{coin_select_bnb, CoinSelector, CoinSelectorOpt, WeightedValue},
    miniscript::{Descriptor, DescriptorPublicKey},
    DescriptorExt, KeychainTracker,
};
use bdk_esplora::ureq::{ureq, Client};
use clap::{Parser, Subcommand};
use std::{
    cmp::Reverse,
    io::{self, Write},
    path::PathBuf,
    time::Duration,
};
mod db;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
struct Args {
    #[clap(env = "DESCRIPTOR")]
    descriptor: String,

    #[clap(env = "BDK_DB_DIR", default_value = ".bdk_example_db")]
    db_dir: PathBuf,

    #[clap(env = "CHANGE_DESCRIPTOR")]
    change_descriptor: Option<String>,

    #[clap(env = "BITCOIN_NETWORK", default_value = "signet", parse(try_from_str))]
    network: Network,

    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Scan,
    Sync {
        #[clap(long)]
        unused: bool,
        #[clap(long)]
        unspent: bool,
        #[clap(long)]
        all: bool,
    },
    Address {
        #[clap(subcommand)]
        addr_cmd: AddressCmd,
    },
    Balance,
    Txo {
        #[clap(subcommand)]
        utxo_cmd: TxoCmd,
    },
    Send {
        value: u64,
        #[clap(parse(try_from_str))]
        address: Address,
        #[clap(parse(try_from_str), short, default_value = "largest-first")]
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

#[derive(Subcommand, Debug)]
pub enum AddressCmd {
    Next,
    New,
    List {
        #[clap(long)]
        change: bool,
    },
    Index,
}

#[derive(Subcommand, Debug)]
pub enum TxoCmd {
    List,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq, bincode::Encode, bincode::Decode)]
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;

    let mut tracker = KeychainTracker::default();
    let mut chain = ChainGraph::default();
    tracker.add_keychain(Keychain::External, descriptor);

    let internal = args
        .change_descriptor
        .map(|descriptor| Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &descriptor))
        .transpose()?;

    let change_keychain = if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        tracker.add_keychain(Keychain::Internal, internal_descriptor);
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

    let mut db = db::Db::load(args.db_dir.as_path(), &mut chain, &mut tracker)?;
    let mut client = Client::new(ureq::Agent::new(), esplora_url);

    match args.command {
        Commands::Scan => {
            client.parallel_requests = 5;
            let stop_gap = 10;

            let spk_iterators = tracker
                .start_wallet_scan()
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
                    chain.chain().checkpoints().clone(),
                )
                .context("scanning the blockchain")?;
            eprintln!();

            db.apply_wallet_scan(&mut chain, &mut tracker, wallet_scan)?;
        }
        Commands::Sync {
            mut unused,
            mut unspent,
            all,
        } => {
            if !(all || unused || unspent) {
                unused = true;
                unspent = true;
            } else if all {
                unused = false;
                unspent = false
            }
            let mut spks = vec![];
            if unused {
                spks.extend(tracker.iter_unused().map(|(index, script)| {
                    eprintln!("Checking if address at {:?} has been used", index);
                    script.clone()
                }));
            }

            if all {
                spks.extend(tracker.script_pubkeys().iter().map(|(index, script)| {
                    eprintln!("scanning {:?}", index);
                    script.clone()
                }));
            }

            if unspent {
                spks.extend(tracker.iter_unspent(chain.chain(), chain.graph()).map(
                    |(_index, ftxout)| {
                        eprintln!("checking if {} has been spent", ftxout.outpoint);
                        ftxout.txout.script_pubkey
                    },
                ));
            }

            let update = client
                .spk_scan(spks.into_iter(), chain.chain().checkpoints().clone())
                .context("scanning the blockchain")?;

            db.apply_wallet_sync(&mut chain, &mut tracker, update)?;
        }
        Commands::Address { addr_cmd } => {
            let new_address = match addr_cmd {
                AddressCmd::Next => Some(tracker.derive_next_unused(Keychain::External)),
                AddressCmd::New => Some(tracker.derive_new(Keychain::External)),
                _ => None,
            };

            if let Some((index, spk)) = new_address {
                let spk = spk.clone();
                // update database since we're about to give out a new address
                db.set_derivation_indicies(&tracker)?;
                let address = Address::from_script(&spk, args.network)
                    .expect("should always be able to derive address");
                eprintln!("This is the address at index {}", index);
                println!("{}", address);
            }

            match addr_cmd {
                AddressCmd::Next | AddressCmd::New => { /* covered */ }
                AddressCmd::Index => {
                    for (keychain, derivation_index) in tracker.derivation_indicies() {
                        println!("{}: {}", keychain, derivation_index);
                    }
                }
                AddressCmd::List { change } => {
                    let target_keychain = match change {
                        true => change_keychain,
                        false => Keychain::External,
                    };
                    for (index, spk) in tracker.script_pubkeys() {
                        if index.0 == target_keychain {
                            let address = Address::from_script(spk, args.network)
                                .expect("should always be able to derive address");
                            println!("{} used:{}", address, tracker.is_used(*index));
                        }
                    }
                }
            }
        }
        Commands::Balance => {
            let (confirmed, unconfirmed) = tracker.iter_unspent(chain.chain(), chain.graph()).fold(
                (0, 0),
                |(confirmed, unconfirmed), ((keychain, _), utxo)| {
                    if utxo.chain_index.is_confirmed() || keychain == Keychain::Internal {
                        (confirmed + utxo.txout.value, unconfirmed)
                    } else {
                        (confirmed, unconfirmed + utxo.txout.value)
                    }
                },
            );

            println!("confirmed: {}", confirmed);
            println!("unconfirmed: {}", unconfirmed);
        }
        Commands::Txo { utxo_cmd } => match utxo_cmd {
            TxoCmd::List => {
                for (spk_index, txout) in tracker.iter_txout().filter_map(|(spk_i, op, _txout)| {
                    chain
                        .chain()
                        .full_txout(chain.graph(), op)
                        .map(|utxo| (spk_i, utxo))
                }) {
                    let script = tracker.spk_at_index(spk_index).unwrap();
                    let address = Address::from_script(script, args.network).unwrap();

                    println!(
                        "{:?} {} {} {} spent:{:?}",
                        spk_index, txout.txout.value, txout.outpoint, address, txout.spent_by
                    )
                }
            }
        },
        Commands::Send {
            value,
            address,
            coin_select,
        } => {
            use bdk_core::miniscript::plan::*;
            let assets = Assets {
                keys: keymap.iter().map(|(pk, _)| pk.clone()).collect(),
                ..Default::default()
            };

            let mut candidates = tracker
                .iter_unspent(chain.chain(), chain.graph())
                .filter_map(|(_, utxo)| {
                    Some((tracker.index_of_spk(&utxo.txout.script_pubkey)?, utxo))
                })
                .filter_map(|((keychain, index), utxo)| {
                    Some((
                        tracker
                            .descriptor(keychain)
                            .at_derivation_index(index)
                            .plan_satisfaction(&assets)?,
                        utxo,
                    ))
                })
                .collect::<Vec<_>>();

            // apply coin selection algorithm
            match coin_select {
                CoinSelectionAlgo::LargestFirst => {
                    candidates.sort_by_key(|(_, utxo)| Reverse(utxo.txout.value))
                }
                CoinSelectionAlgo::SmallestFirst => {
                    candidates.sort_by_key(|(_, utxo)| utxo.txout.value)
                }
                CoinSelectionAlgo::OldestFirst => {
                    candidates.sort_by_key(|(_, utxo)| utxo.chain_index)
                }
                CoinSelectionAlgo::NewestFirst => {
                    candidates.sort_by_key(|(_, utxo)| Reverse(utxo.chain_index))
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

            let (change_index, change_script) = {
                let (index, script) = tracker.derive_next_unused(change_keychain);
                (index, script.clone())
            };
            let change_plan = tracker
                .descriptor(change_keychain)
                .at_derivation_index(change_index)
                .plan_satisfaction(&assets)
                .expect("failed to obtain change plan");

            let mut change_output = TxOut {
                value: 0,
                script_pubkey: change_script,
            };

            let cs_opts = CoinSelectorOpt {
                target_feerate: 0.5,
                min_drain_value: tracker.descriptor(change_keychain).dust_value(),
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
                lock_time: chain
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
                        &secp,
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

            eprintln!("broadcasting transactions..");
            match client
                .broadcast(&transaction)
                .context("broadcasting transaction")
            {
                Ok(_) => println!("{}", transaction.txid()),
                Err(e) => eprintln!(
                    "Failed to broadcast transaction:\n{}\nError:{}",
                    consensus::serialize(&transaction).to_hex(),
                    e
                ),
            }
        }
    }
    Ok(())
}
