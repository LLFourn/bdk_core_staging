use anyhow::{anyhow, Context};
use bdk_core::{
    bitcoin::{
        consensus,
        hashes::hex::ToHex,
        secp256k1::Secp256k1,
        util::sighash::{Prevouts, SighashCache},
        Address, LockTime, Network, Sequence, Transaction, TxIn, TxOut,
    },
    coin_select::{CoinSelector, CoinSelectorOpt, WeightedCandidate, TXIN_BASE_WEIGHT},
    miniscript::{Descriptor, DescriptorPublicKey},
    ApplyResult, DescriptorExt, KeychainTracker, SparseChain,
};
use bdk_esplora::ureq::{ureq, Client};
use clap::{Parser, Subcommand};
use std::cmp::Reverse;

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
#[clap(propagate_version = true)]
struct Args {
    #[clap(env = "DESCRIPTOR")]
    descriptor: String,

    #[clap(env = "CHANGE_DESCRIPTOR")]
    change_descriptor: Option<String>,

    #[clap(env = "BITCOIN_NETWORK", default_value = "signet", parse(try_from_str))]
    network: Network,

    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
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
}

#[derive(Subcommand, Debug)]
pub enum TxoCmd {
    List,
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum Keychain {
    External,
    Internal,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;

    let mut tracker = KeychainTracker::default();
    let mut chain = SparseChain::default();
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

    let mut client = Client::new(ureq::Agent::new(), esplora_url);
    client.parallel_requests = 5;

    fully_sync(&client, &mut tracker, &mut chain)?;

    match args.command {
        Commands::Address { addr_cmd } => {
            let new_address = match addr_cmd {
                AddressCmd::Next => Some(tracker.derive_next_unused(Keychain::External)),
                AddressCmd::New => Some(tracker.derive_new(Keychain::External)),
                _ => None,
            };

            if let Some((index, spk)) = new_address {
                let address = Address::from_script(&spk, args.network)
                    .expect("should always be able to derive address");
                eprintln!("This is the address at index {}", index);
                println!("{}", address);
            }

            match addr_cmd {
                AddressCmd::Next | AddressCmd::New => { /* covered */ }
                AddressCmd::List { change } => {
                    let target_keychain = match change {
                        true => change_keychain,
                        false => Keychain::External,
                    };
                    for (index, spk) in tracker.script_pubkeys() {
                        if index.0 == target_keychain {
                            let address = Address::from_script(&spk, args.network)
                                .expect("should always be able to derive address");
                            println!("{} used:{}", address, tracker.is_used(*index));
                        }
                    }
                }
            }
        }
        Commands::Balance => {
            let (confirmed, unconfirmed) = tracker.iter_unspent_full(&chain).fold(
                (0, 0),
                |(confirmed, unconfirmed), (spk_index, utxo)| {
                    if utxo.confirmed_at.is_some() || spk_index.0 == Keychain::Internal {
                        (confirmed + utxo.value, unconfirmed)
                    } else {
                        (confirmed, unconfirmed + utxo.value)
                    }
                },
            );

            println!("confirmed: {}", confirmed);
            println!("unconfirmed: {}", unconfirmed);
        }
        Commands::Txo { utxo_cmd } => match utxo_cmd {
            TxoCmd::List => {
                for (spk_index, txout) in tracker.iter_txout_full(&chain) {
                    let script = tracker.spk_at_index(spk_index).unwrap();
                    let address = Address::from_script(script, args.network).unwrap();

                    println!(
                        "{:?} {} {} {} spent:{:?}",
                        spk_index, txout.value, txout.outpoint, address, txout.spent_by
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
                .iter_unspent_full(&chain)
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
                    candidates.sort_by_key(|(_, utxo)| Reverse(utxo.value))
                }
                CoinSelectionAlgo::SmallestFirst => candidates.sort_by_key(|(_, utxo)| utxo.value),
                CoinSelectionAlgo::OldestFirst => candidates.sort_by_key(|(_, utxo)| {
                    utxo.confirmed_at
                        .map(|utxo| utxo.height)
                        .unwrap_or(u32::MAX)
                }),
                CoinSelectionAlgo::NewestFirst => candidates.sort_by_key(|(_, utxo)| {
                    Reverse(
                        utxo.confirmed_at
                            .map(|utxo| utxo.height)
                            .unwrap_or(u32::MAX),
                    )
                }),
                CoinSelectionAlgo::BranchAndBound => todo!(),
            }

            // turn the txos we chose into a weight and value
            let wv_candidates = candidates
                .iter()
                .map(|(plan, utxo)| WeightedCandidate {
                    value: utxo.value,
                    base_weight: TXIN_BASE_WEIGHT,
                    satisfaction_weight: plan.expected_weight() as u32,
                    is_segwit: plan.witness_version().is_some(),
                })
                .collect();

            let mut outputs = vec![TxOut {
                value,
                script_pubkey: address.script_pubkey(),
            }];

            let mut change_output = TxOut {
                value: 0,
                script_pubkey: tracker.derive_next_unused(change_keychain).1.clone(),
            };

            let coin_selector_opts = CoinSelectorOpt {
                target_feerate: 0.5,
                ..CoinSelectorOpt::fund_outputs(&outputs, &change_output)
            };

            // TODO: How can we make it easy to shuffle in order of inputs and outputs here?
            // apply coin selection by saying we need to fund these outputs
            let mut coin_selector = CoinSelector::new(wv_candidates, &coin_selector_opts);

            // just select coins in the order provided until we have enough
            let selection = coin_selector.select_until_finished()?;

            // get the selected utxos
            let selected_txos = selection.iter_selected(&candidates).collect::<Vec<_>>();

            if selection.use_drain
                && selection.excess >= tracker.descriptor(change_keychain).dust_value()
            {
                change_output.value = selection.excess;
                // if the selection tells us to use change and the change value is sufficient we add it as an output
                outputs.push(change_output)
            }

            let mut transaction = Transaction {
                version: 0x02,
                lock_time: chain
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
                .map(|(_, utxo)| TxOut::from(utxo.clone()))
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

pub fn fully_sync(
    client: &Client,
    tracker: &mut KeychainTracker<Keychain>,
    chain: &mut SparseChain,
) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
    let mut active_indexes = vec![];
    for (keychain, descriptor) in tracker.iter_keychains(..) {
        eprint!("scanning {:?} addresses indexes ", keychain);
        let (last_active_index, checkpoint) = client
            .fetch_new_checkpoint(
                descriptor
                    .iter_all_scripts()
                    .enumerate()
                    .map(|(i, script)| (i as u32, script))
                    .inspect(|(i, _)| {
                        use std::io::{self, Write};
                        eprint!("{} ", i);
                        let _ = io::stdout().flush();
                    }),
                2,
                chain.iter_checkpoints(..).rev(),
            )
            .context("fetching transactions")?;

        match chain.apply_checkpoint(checkpoint) {
            ApplyResult::Ok => eprintln!("success! ({}ms)", start.elapsed().as_millis()),
            ApplyResult::Stale(_reason) => {
                unreachable!("we are the only ones accessing the tracker")
            }
            ApplyResult::Inconsistent {
                txid,
                conflicts_with,
            } => {
                return Err(anyhow!(
                    "blockchain backend returned conflicting info: {} conflicts with {}",
                    txid,
                    conflicts_with,
                ))
            }
        }

        if let Some(last_active_index) = last_active_index {
            active_indexes.push((keychain, last_active_index));
        }
    }

    for (keychain, active_index) in active_indexes {
        tracker.derive_spks(keychain, active_index);
    }

    tracker.sync(&chain);

    Ok(())
}
