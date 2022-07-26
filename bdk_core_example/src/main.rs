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
use bdk_core::ApplyResult;
use bdk_core::DescriptorExt;
use bdk_core::ScriptTracker;
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

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum Keychain {
    Internal,
    External,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let secp = Secp256k1::default();
    let (descriptor, mut keymap) =
        Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &args.descriptor)?;
    // external tracker
    let mut tracker = ScriptTracker::<(Keychain, u32)>::default();

    let mut descriptors = vec![descriptor];

    let internal = args
        .change_descriptor
        .map(|descriptor| Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, &descriptor))
        .transpose()?;

    // // trackers are in a Vec<DescriptorTracker> which has methods attached to it via extension traits
    // // 0 -> external tacker
    // // 1 -> internal tracker
    // let mut trackers = vec![tracker];
    if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        descriptors.push(internal_descriptor);
    }

    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3000",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let mut client = Client::new(ureq::Agent::new(), esplora_url);
    client.parallel_requests = 5;

    for keychain in [Keychain::Internal, Keychain::External] {
        fully_sync_keychain(
            keychain,
            &descriptors[keychain as usize],
            &client,
            &mut tracker,
        )?;
    }

    match args.command {
        Commands::Address { addr_cmd } => {
            let new_address = match addr_cmd {
                AddressCmd::Next => Some(tracker.keychain_derive_next_unused(
                    Keychain::External,
                    &descriptors[Keychain::External as usize],
                )),
                AddressCmd::New => Some(tracker.keychain_derive_new(
                    Keychain::External,
                    &descriptors[Keychain::External as usize],
                )),
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
                        true => Keychain::Internal,
                        false => Keychain::External,
                    };
                    for (index, spk) in tracker.scripts() {
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
            let (confirmed, unconfirmed) =
                tracker
                    .iter_unspent()
                    .fold((0, 0), |(confirmed, unconfirmed), utxo| {
                        if utxo.confirmed_at.is_some() || utxo.spk_index.0 == Keychain::Internal {
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
                for txout in tracker.iter_txout() {
                    let script = tracker.script_at_index(&txout.spk_index).unwrap();
                    let address = Address::from_script(script, args.network).unwrap();

                    println!(
                        "{:?} {} {} {} spent:{:?}",
                        txout.spk_index, txout.value, txout.outpoint, address, txout.spent_by
                    )
                }
            }
        },
        Commands::Send {
            value,
            address,
            coin_select,
        } => {
            let mut candidates = tracker.iter_unspent().collect::<Vec<_>>();

            // apply coin selection algorithm
            match coin_select {
                CoinSelectionAlgo::LargestFirst => {
                    candidates.sort_by_key(|utxo| Reverse(utxo.value))
                }
                CoinSelectionAlgo::SmallestFirst => candidates.sort_by_key(|utxo| utxo.value),
                CoinSelectionAlgo::OldestFirst => candidates.sort_by_key(|utxo| {
                    utxo.confirmed_at
                        .map(|utxo| utxo.height)
                        .unwrap_or(u32::MAX)
                }),
                CoinSelectionAlgo::NewestFirst => candidates.sort_by_key(|utxo| {
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
                .map(|utxo| WeightedValue {
                    value: utxo.value,
                    weight: descriptors[utxo.spk_index.0 as usize]
                        .max_satisfaction_weight()
                        .unwrap() as u32,
                })
                .collect();

            let mut outputs = vec![TxOut {
                value,
                script_pubkey: address.script_pubkey(),
            }];

            let mut change_output = TxOut {
                value: 0,
                script_pubkey: tracker
                    .keychain_derive_next_unused(
                        Keychain::Internal,
                        &descriptors[Keychain::Internal as usize],
                    )
                    .1
                    .clone(),
            };

            // apply coin selection by saying we need to fund these outputs
            let mut coin_selector = CoinSelector::new(
                wv_candidates,
                CoinSelectorOpt::fund_outputs(&outputs, &change_output),
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

            dbg!(&selection);

            // get the selected utxos
            let selected_txos = selection.apply_selection(&candidates).collect::<Vec<_>>();

            if selection.use_change
                && selection.excess >= descriptors[Keychain::Internal as usize].dust_value()
            {
                change_output.value = selection.excess;
                // if the selection tells us to use change and the change value is sufficient we add it as an output
                outputs.push(change_output)
            }

            // TODO: How can we make it easy to shuffle in order of inputs and outputs here?

            // With the selected utxos and outputs create our PSBT and return the descriptor for the
            // spk of each selected input. We will need the descriptors to build the witness
            // properly with ".satisfy".
            let (mut psbt, definite_descriptors) = tracker.create_psbt(
                selected_txos.iter().map(|txo| txo.outpoint),
                outputs,
                |(keychain, derivation_index)| {
                    Some(descriptors[keychain as usize].at_derivation_index(derivation_index))
                },
            );

            let cache_tx = psbt.unsigned_tx.clone();
            let mut sighash_cache = SighashCache::new(&cache_tx);

            // Go through the selected inputs and try and sign them with the secret keys we got out
            // of the descriptor.
            //
            // TODO: It would be great to be able to make this easier.
            // I think the solution lies in integrating the "policy" module of existing bdk.
            for (input_index, selected_txo) in selected_txos.iter().enumerate() {
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

pub fn fully_sync_keychain(
    keychain: Keychain,
    descriptor: &Descriptor<DescriptorPublicKey>,
    client: &Client,
    tracker: &mut ScriptTracker<(Keychain, u32)>,
) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
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
            tracker.iter_checkpoints(),
        )
        .context("fetching transactions")?;
    if let Some(last_active_index) = last_active_index {
        tracker.keychain_derive_scripts(keychain, descriptor, last_active_index);
    }
    match tracker.apply_checkpoint(checkpoint) {
        ApplyResult::Ok => eprintln!("success! ({}ms)", start.elapsed().as_millis()),
        ApplyResult::Stale => unreachable!("we are the only ones accessing the tracker"),
        ApplyResult::Inconsistent {
            txid,
            conflicts_with,
            at_checkpoint,
        } => {
            return Err(anyhow!(
                "blockchain backend returned conflicting info: {} conflicts with {} at {:?}",
                txid,
                conflicts_with,
                at_checkpoint
            ))
        }
    }
    Ok(())
}
