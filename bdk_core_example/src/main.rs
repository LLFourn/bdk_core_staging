use anyhow::{anyhow, Context};
use bdk_core::bitcoin::consensus;
use bdk_core::bitcoin::hashes::hex::ToHex;
use bdk_core::bitcoin::secp256k1::Secp256k1;
use bdk_core::bitcoin::util::sighash::SighashCache;
use bdk_core::bitcoin::Address;
use bdk_core::bitcoin::Network;
use bdk_core::bitcoin::TxIn;
use bdk_core::bitcoin::TxOut;
use bdk_core::coin_select::WeightedValue;
use bdk_core::coin_select::{CoinSelector, CoinSelectorOpt};
use bdk_core::miniscript::psbt::PsbtInputSatisfier;
use bdk_core::miniscript::Descriptor;
use bdk_core::miniscript::DescriptorPublicKey;
use bdk_core::{DescriptorTracker, MultiTracker};
use bdk_esplora::ureq::{ureq, Client};
use clap::Parser;
use clap::Subcommand;
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

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;
    // external tracker
    let tracker = DescriptorTracker::new(descriptor);
    let change_tracker = args
        .change_descriptor
        .map(|descriptor| {
            let (descriptor, _key_map) =
                Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &descriptor)?;
            Ok::<_, bdk_core::miniscript::Error>(DescriptorTracker::new(descriptor))
        })
        .transpose()?;

    // trackers are in a Vec<DescriptorTracker> which has methods attached to it via extension traits
    // 0 -> external tacker
    // 1 -> internal tracker
    let mut trackers = vec![tracker];
    if let Some(change_tracker) = change_tracker {
        trackers.push(change_tracker);
    }

    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3000",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let mut client = Client::new(ureq::Agent::new(), esplora_url);
    client.parallel_requests = 5;

    for (i, tracker) in trackers.iter_mut().enumerate() {
        fully_sync_tracker(
            match i {
                1 => "internal",
                _ => "external",
            },
            &client,
            tracker,
        )?;
    }

    match args.command {
        Commands::Address { addr_cmd } => {
            let new_address = match addr_cmd {
                AddressCmd::Next => Some(trackers[0].derive_next_unused()),
                AddressCmd::New => Some(trackers[0].derive_new()),
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
                    let tracker = if change {
                        trackers
                            .get(1)
                            .ok_or(anyhow!("you havent set a change descriptor"))?
                    } else {
                        &trackers[0]
                    };

                    for (i, spk) in tracker.iter_derived_scripts().enumerate() {
                        let address = Address::from_script(&spk, args.network)
                            .expect("should always be able to derive address");
                        println!("{} used:{}", address, tracker.is_used(i as u32));
                    }
                }
            }
        }
        Commands::Balance => {
            let utxos = trackers
                .iter_unspent()
                .map(|(tracker_id, utxo)| (tracker_id == 1, utxo));
            let (confirmed, unconfirmed) =
                utxos.fold((0, 0), |(confirmed, unconfirmed), (is_change, utxo)| {
                    if utxo.confirmed_at.is_some() || is_change {
                        (confirmed + utxo.value, unconfirmed)
                    } else {
                        (confirmed, unconfirmed + utxo.value)
                    }
                });

            println!("confirmed: {}", confirmed);
            println!("unconfirmed: {}", unconfirmed);
        }
        Commands::Txo { utxo_cmd } => match utxo_cmd {
            TxoCmd::List => {
                for (tracker_index, txout) in trackers.iter_txout() {
                    let script = trackers[tracker_index]
                        .script_at_index(txout.derivation_index)
                        .unwrap();
                    let address = Address::from_script(script, args.network).unwrap();

                    println!(
                        "keychain:{} {} {} {} spent:{:?}",
                        tracker_index, txout.value, txout.outpoint, address, txout.spent_by
                    )
                }
            }
        },
        Commands::Send {
            value,
            address,
            coin_select,
        } => {
            let mut candidates = trackers.iter_unspent().collect::<Vec<_>>();

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
                .map(|(tracker_index, utxo)| WeightedValue {
                    value: utxo.value,
                    weight: trackers[*tracker_index].max_satisfaction_weight(),
                })
                .collect();

            let mut outputs = vec![TxOut {
                value,
                script_pubkey: address.script_pubkey(),
            }];

            // apply coin selection by saying we need to fund these outputs
            let mut coin_selector = CoinSelector::new(
                wv_candidates,
                CoinSelectorOpt::fund_outputs(&outputs, trackers[1].max_satisfaction_weight()),
            );

            // just select coins in the order provided until we have enough
            let selection = match coin_selector.select_until_finished() {
                Some(selection) => selection,
                None => {
                    return Err(anyhow!(
                        "Insufficient funds. Needed {} had {}",
                        value,
                        coin_selector.current_value()
                    ))
                }
            };

            // get the selected utxos
            let selected_txos = selection.apply_selection(&candidates).collect::<Vec<_>>();

            let change_tracker = trackers.last_mut().unwrap();

            if selection.use_change && selection.excess >= change_tracker.dust_value() {
                // if the selection tells us to use change and the change value is sufficient we add it as an output
                outputs.push(TxOut {
                    value: selection.excess,
                    script_pubkey: change_tracker.derive_new().1.clone(),
                })
            }

            // TODO: How can we make it easy to shuffle in order of inputs and outputs here?

            // With the selected utxos and outputs create our PSBT and return the descriptor for the
            // spk of each selected input. We will need the descriptors to build the witness
            // properly with ".satisfy".
            let (mut psbt, definite_descriptors) =
                trackers.create_psbt(selected_txos.iter().map(|(_, txo)| txo.outpoint), outputs);

            let cache_tx = psbt.unsigned_tx.clone();
            let mut sighash_cache = SighashCache::new(&cache_tx);

            // Go through the selected inputs and try and sign them with the secret keys we got out
            // of the descriptor.
            //
            // TODO: It would be great to be able to make this easier.
            // I think the solution lies in integrating the "policy" module of existing bdk.
            for (input_index, (_, selected_txo)) in selected_txos.iter().enumerate() {
                if let Some(definite_descriptor) = definite_descriptors.get(&input_index) {
                    eprintln!("signing input {}", selected_txo.outpoint);
                    for (pk, sk) in &keymap {
                        let signed = bdk_core::sign::sign_with_descriptor_sk(
                            sk,
                            &mut psbt,
                            &mut sighash_cache,
                            input_index,
                            &secp,
                        )?;
                        if !signed {
                            eprintln!("the secret key for {} failed to sign anything (maybe we haven't implmented that kind of signing yet)", pk);
                        }
                    }
                    let mut tmp_input = TxIn::default();
                    // TODO: try to satisfy with time locks as well
                    match definite_descriptor
                        .satisfy(&mut tmp_input, PsbtInputSatisfier::new(&psbt, input_index))
                    {
                        Ok(_) => {
                            let psbt_input = &mut psbt.inputs[input_index];
                            psbt_input.final_script_sig = Some(tmp_input.script_sig);
                            psbt_input.final_script_witness = Some(tmp_input.witness);
                        }
                        Err(e) => {
                            return Err(anyhow!(
                                "unable to satsify spending conditions of {}: {}",
                                selected_txo.outpoint,
                                e
                            ))
                        }
                    }
                }
            }

            let signed_tx = psbt.extract_tx();
            eprintln!("broadcasting transactions..");
            match client
                .broadcast(&signed_tx)
                .context("broadcasting transaction")
            {
                Ok(_) => println!("{}", signed_tx.txid()),
                Err(e) => eprintln!(
                    "Failed to broadcast transaction:\n{}\nError:{}",
                    consensus::serialize(&signed_tx).to_hex(),
                    e
                ),
            }
        }
    }
    Ok(())
}

pub fn fully_sync_tracker(
    name: &str,
    client: &Client,
    tracker: &mut DescriptorTracker,
) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
    eprint!("scanning {} addresses indexes ", name);
    let update = client
        .fetch_related_transactions(
            tracker
                .iter_scripts()
                .enumerate()
                .map(|(i, script)| (i as u32, script))
                .inspect(|(i, _)| {
                    use std::io::{self, Write};
                    eprint!("{} ", i);
                    let _ = io::stdout().flush();
                }),
            2,
            core::iter::empty(),
        )
        .context("fetching transactions")?;
    eprintln!("success! ({}ms)", start.elapsed().as_millis());
    tracker.apply_update(update);
    Ok(())
}
