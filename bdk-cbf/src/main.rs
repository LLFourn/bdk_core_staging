use anyhow::anyhow;
use bdk_cbf::CbfNode;
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
use bdk_core::BlockId;
use bdk_core::DescriptorExt;
use bdk_core::KeychainTracker;
use bdk_core::SparseChain;
use bitcoin::Script;
use std::cmp::Reverse;
use std::str::FromStr;
use structopt::StructOpt;

/// BDK-CBF Client Application
///
/// This is a light weight command line tool to sync any descriptor with BIP157
/// compact block filters.
#[derive(StructOpt)]
#[structopt(version = option_env ! ("CARGO_PKG_VERSION").unwrap_or("unknown"),
author = option_env ! ("CARGO_PKG_AUTHORS").unwrap_or(""))]
#[structopt(name = "bdk-cbf")]
struct Args {
    /// Sets the main descriptor to watch for
    #[structopt(short = "d", long = "descriptor")]
    descriptor: String,

    /// Sets azn optional change descriptor to watch for
    #[structopt(short = "c", long = "change-descriptor")]
    change_descriptor: Option<String>,

    /// Sets the network
    #[structopt(short, long, default_value = "signet", possible_values = &["bitcoin", "testnet", "signet", "regtest"])]
    network: Network,

    /// Sets a block height to start syncing from
    #[structopt(short, long, default_value = "0")]
    birthday: u32,

    #[structopt(subcommand)]
    command: Commands,
}

#[derive(StructOpt, Debug)]
enum Commands {
    /// Get new addresses
    Address {
        #[structopt(subcommand)]
        addr_cmd: AddressCmd,
    },
    /// Get the descriptor balance
    Balance,
    /// Get available utxos
    UTxo,
    /// Send transaction
    Send {
        /// Adds a recipient to the transaction
        #[structopt(name = "ADDRESS:SAT", long = "to", required = true, parse(try_from_str = parse_recipient))]
        recipients: Vec<(Script, u64)>,
        #[structopt(short, default_value = "largest-first")]
        /// Specify coinselection algorithm
        coin_select: CoinSelectionAlgo,
    },
}

/// Parse the recipient (Address,Amount) argument from cli input
pub(crate) fn parse_recipient(s: &str) -> Result<(Script, u64), String> {
    let parts: Vec<_> = s.split(':').collect();
    if parts.len() != 2 {
        return Err("Invalid format".to_string());
    }
    let addr = Address::from_str(parts[0]).map_err(|e| e.to_string())?;
    let val = u64::from_str(parts[1]).map_err(|e| e.to_string())?;

    Ok((addr.script_pubkey(), val))
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

impl FromStr for CoinSelectionAlgo {
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

#[derive(StructOpt, Debug)]
pub enum AddressCmd {
    /// Get the next unused address
    Next,
    /// Get a new unused address
    New,
    /// Get a list of unused addresses
    List {
        #[structopt(long)]
        /// get a new change address
        change: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialOrd, Ord, PartialEq, Eq)]
pub enum Keychain {
    External,
    Internal,
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::init();
    let args = Args::from_args();
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

    if let Some((internal_descriptor, internal_keymap)) = internal {
        keymap.extend(internal_keymap);
        tracker.add_keychain(Keychain::Internal, internal_descriptor);
    }

    let cbf_node = sync_chain(args.birthday, args.network, &mut tracker, &mut chain)?;

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
                        true => Keychain::Internal,
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
        Commands::UTxo => {
            for (spk_index, txout) in tracker.iter_txout_full(&chain) {
                let script = tracker.spk_at_index(spk_index).unwrap();
                let address = Address::from_script(script, args.network).unwrap();

                println!(
                    "{:?} {} {} {} spent:{:?}",
                    spk_index, txout.value, txout.outpoint, address, txout.spent_by
                )
            }
        }
        Commands::Send {
            recipients,
            coin_select,
        } => {
            let mut candidates = tracker.iter_unspent_full(&chain).collect::<Vec<_>>();

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
                .map(|(spk_index, utxo)| WeightedValue {
                    value: utxo.value,
                    weight: tracker
                        .descriptor(spk_index.0)
                        .max_satisfaction_weight()
                        .unwrap() as u32,
                })
                .collect();

            let mut outputs = recipients
                .iter()
                .map(|(script, value)| TxOut {
                    value: *value,
                    script_pubkey: script.clone(),
                })
                .collect::<Vec<_>>();

            let to_send = outputs.iter().fold(0, |sum, output| sum + output.value);

            let mut change_output = TxOut {
                value: 0,
                script_pubkey: tracker.derive_next_unused(Keychain::Internal).1.clone(),
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
                        to_send,
                        coin_selector.current_value()
                    ))
                }
            };

            // get the selected utxos
            let selected_txos = selection.apply_selection(&candidates).collect::<Vec<_>>();

            if selection.use_change
                && selection.excess >= tracker.descriptor(Keychain::Internal).dust_value()
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
                selected_txos.iter().map(|(_, txo)| txo.outpoint),
                outputs,
                &chain,
            );

            let cache_tx = psbt.unsigned_tx.clone();
            let mut sighash_cache = SighashCache::new(&cache_tx);

            // Go through the selected inputs and try and sign them with the secret keys we got out
            // of the descriptor.
            //
            // TODO: It would be great to be able to make this easier.
            // I think the solution lies in integrating the "policy" module of existing bdk.
            for (input_index, (spk_index, selected_txo)) in selected_txos.iter().enumerate() {
                if let Some(definite_descriptor) = definite_descriptors.get(&input_index) {
                    eprintln!(
                        "signing input {} derived at {:?}",
                        selected_txo.outpoint, spk_index
                    );
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

            #[allow(unused_variables)]
            let signed_tx = psbt.extract_tx();
            println!("broadcasting transactions..");
            cbf_node.braodcast(signed_tx);
        }
    }
    cbf_node.shutdown();
    Ok(())
}

pub fn sync_chain(
    birthday: u32,
    network: Network,
    tracker: &mut KeychainTracker<Keychain>,
    chain: &mut SparseChain,
) -> anyhow::Result<CbfNode> {
    use bdk_cbf::nakamoto_client::Event;

    let mut scripts = vec![];

    for keychain in tracker.iter_keychains(..).collect::<Vec<_>>() {
        tracker.derive_spks(keychain, 99);
        scripts.extend(tracker.script_pubkeys().values().cloned())
    }

    let node = CbfNode::new(network, scripts);
    node.scan(birthday);

    loop {
        match node.get_next_event()? {
            Event::BlockConnected { hash, height, .. } => {
                eprintln!("block connected: {} {}", hash, height)
            }
            Event::BlockDisconnected { hash, height, .. } => chain.disconnect_block(BlockId {
                hash,
                height: height as u32,
            }),
            Event::BlockMatched {
                hash,
                header,
                height,
                transactions,
            } => {
                let block_id = BlockId {
                    hash,
                    height: height as u32,
                };
                eprintln!(
                    "found block {:?} with {} transactions",
                    block_id,
                    transactions.len()
                );
                match chain.apply_block_txs(block_id, header.time as u64, transactions) {
                    bdk_core::ApplyResult::Ok => eprintln!("successfully applied block!"),
                    bdk_core::ApplyResult::Stale(_) => eprintln!("block was stale!"),
                    bdk_core::ApplyResult::Inconsistent {
                        txid,
                        conflicts_with,
                    } => eprintln!("block had inconsistent txs: {}, {}", txid, conflicts_with),
                }
            }
            Event::FeeEstimated { .. } => eprintln!("We never estimated a fee!"),
            Event::FilterProcessed { .. } => { /* this happens every time it matches against the filter during scanning */
            }
            Event::TxStatusChanged { txid, status } => {
                eprintln!("tx {} status changed to {}", txid, status)
            }
            Event::Synced { height, tip } => {
                if height == tip {
                    break;
                }
            }
            _ => { /* ignore */ }
        }
    }

    tracker.sync(&chain);

    Ok(node)
}
