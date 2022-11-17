use anyhow::{anyhow, Context};
use bdk_core::{
    bitcoin::{
        consensus,
        hashes::hex::ToHex,
        secp256k1::{Secp256k1, Signing, Verification},
        util::psbt,
        util::sighash::{Prevouts, SighashCache},
        Address, LockTime, Network, PackedLockTime, Sequence, Transaction, TxIn, TxOut, Witness,
    },
    coin_select::{coin_select_bnb, CoinSelector, CoinSelectorOpt, WeightedValue},
    descriptor_into_script_iter,
    miniscript::{
        descriptor::KeyMap, plan::RequiredSig, DefiniteDescriptorKey, Descriptor,
        DescriptorPublicKey,
    },
    ChainGraph, ChainIndex, DescriptorExt, KeychainTracker, TimestampedChainGraph,
};
use bdk_esplora::ureq::{ureq, Client};
use clap::{Parser, Subcommand};
use std::{cmp::Reverse, time::Duration};

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
            use bdk_core::miniscript::psbt::*;

            let assets = keymap.iter().map(|(pk, _)| pk.clone()).collect::<Assets>();

            let mut candidates = tracker
                .iter_unspent(chain.chain(), chain.graph())
                .filter_map(|(_, utxo)| {
                    Some((tracker.index_of_spk(&utxo.txout.script_pubkey)?, utxo))
                })
                .filter_map(|((keychain, index), utxo)| {
                    let derived_desc = tracker.descriptor(keychain).at_derivation_index(index);
                    Some((derived_desc.get_plan(&assets)?, derived_desc, utxo))
                })
                .collect::<Vec<_>>();

            // apply coin selection algorithm
            match coin_select {
                CoinSelectionAlgo::LargestFirst => {
                    candidates.sort_by_key(|(_, _, utxo)| Reverse(utxo.txout.value))
                }
                CoinSelectionAlgo::SmallestFirst => {
                    candidates.sort_by_key(|(_, _, utxo)| utxo.txout.value)
                }
                CoinSelectionAlgo::OldestFirst => {
                    candidates.sort_by_key(|(_, _, utxo)| utxo.chain_index.height())
                }
                CoinSelectionAlgo::NewestFirst => {
                    candidates.sort_by_key(|(_, _, utxo)| Reverse(utxo.chain_index.height()))
                }
                CoinSelectionAlgo::BranchAndBound => {}
            }

            // turn the txos we chose into a weight and value
            let wv_candidates = candidates
                .iter()
                .map(|(plan, _, utxo)| {
                    WeightedValue::new(
                        utxo.txout.value,
                        plan.satisfaction_weight() as _,
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
                .get_plan(&assets)
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
                    change_plan.satisfaction_weight() as u32,
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

            let transaction = Transaction {
                version: 0x02,
                lock_time: PackedLockTime(0),
                input: selected_txos
                    .iter()
                    .map(|(_, _, utxo)| TxIn {
                        previous_output: utxo.outpoint,
                        sequence: Sequence::ENABLE_RBF_NO_LOCKTIME,
                        ..Default::default()
                    })
                    .collect(),
                output: outputs,
            };
            let mut locktime = chain
                .chain()
                .latest_checkpoint()
                .and_then(|block_id| LockTime::from_height(block_id.height).ok())
                .unwrap_or(LockTime::ZERO);

            let mut psbt = psbt::Psbt::from_unsigned_tx(transaction)?;

            // first set tx values for plan so that we don't change them while signing
            for (i, (plan, desc, utxo)) in selected_txos.iter().enumerate() {
                if let Some(sequence) = plan.relative_timelock {
                    psbt.unsigned_tx.input[i].sequence = sequence
                }
                if let Some(plan_locktime) = plan.absolute_timelock {
                    if plan_locktime > locktime {
                        locktime = plan_locktime
                    }
                }

                psbt.inputs[i].witness_utxo = Some(utxo.txout.clone());
                psbt.update_input_with_descriptor(i, desc)?;
            }
            psbt.unsigned_tx.lock_time = locktime.into();

            for (i, (plan, _, _)) in selected_txos.iter().enumerate() {
                assert!(
                    plan.template.required_preimages().is_empty(),
                    "can't have hash pre-images since we didn't provide any"
                );

                for req in plan.template.required_signatures() {
                    sign_psbt_with_keymap(req, &mut psbt, i, &keymap, &secp)?;
                }

                // TODO: this assumes taproot scripts,
                let stfr = PsbtInputSatisfier::new(&psbt, i);
                match plan.template.try_completing(&stfr) {
                    Some(witness) => {
                        psbt.inputs[i].final_script_witness = Some(Witness::from_vec(witness))
                    }
                    None => {
                        return Err(anyhow!(
                            "we weren't able to complete the plan with our keys"
                        ))
                    }
                }
            }

            eprintln!("broadcasting transactions..");
            let extracted_transaction = psbt.extract_tx();
            match client
                .broadcast(&extracted_transaction)
                .context("broadcasting transaction")
            {
                Ok(_) => println!("{}", extracted_transaction.txid()),
                Err(e) => eprintln!(
                    "Failed to broadcast transaction:\n{}\nError:{}",
                    consensus::serialize(&extracted_transaction).to_hex(),
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
    chain: &mut TimestampedChainGraph,
) -> anyhow::Result<()> {
    let start = std::time::Instant::now();
    let mut active_indexes = vec![];
    let mut update = ChainGraph::default();

    for (keychain, descriptor) in tracker.iter_keychains(..) {
        eprint!("scanning {:?} addresses indexes ", keychain);
        let (last_active_index, keychain_update) = client
            .fetch_new_checkpoint(
                descriptor_into_script_iter(descriptor.clone())
                    .enumerate()
                    .map(|(i, script)| (i as u32, script))
                    .inspect(|(i, _)| {
                        use std::io::{self, Write};
                        eprint!("{} ", i);
                        let _ = io::stdout().flush();
                    }),
                2,
                chain.chain().checkpoints(),
            )
            .context("fetching transactions")?;

        update
            .apply_update(&keychain_update)
            .map_err(|_| anyhow!("the updates for the two keychains were incompatible"))?;

        if let Some(last_active_index) = last_active_index {
            active_indexes.push((keychain, last_active_index));
        }

        eprintln!("success! ({}ms)", start.elapsed().as_millis())
    }

    match chain.apply_update(&update) {
        Result::Ok(_changes) => { /* TODO: print out the changes nicely */ }
        Result::Err(_reason) => {
            unreachable!("we are the only ones accessing the tracker");
        }
    }

    // for tx in update.graph().iter_full_txs() {
    //     graph.insert_tx(tx);
    // }

    for (keychain, active_index) in active_indexes {
        tracker.derive_spks(keychain, active_index);
    }

    tracker.scan(chain.graph());

    Ok(())
}

pub fn sign_psbt_with_keymap(
    required_sig: RequiredSig<'_, DefiniteDescriptorKey>,
    psbt: &mut psbt::Psbt,
    input_index: usize,
    keymap: &KeyMap,
    // auth_data: &mut SatisfactionMaterial,
    secp: &Secp256k1<impl Signing + Verification>,
) -> anyhow::Result<bool> {
    use bdk_core::bitcoin::secp256k1::{KeyPair, Message, PublicKey};
    use bdk_core::bitcoin::{
        util::schnorr::SchnorrSig, util::sighash::SchnorrSighashType, util::taproot, XOnlyPublicKey,
    };
    use bdk_core::miniscript::descriptor::DescriptorSecretKey;

    let prevouts = psbt
        .inputs
        .iter()
        .map(|i| {
            i.witness_utxo
                .clone()
                .expect("witness_utxo must be present")
        })
        .collect::<Vec<_>>();
    let prevouts = Prevouts::All(&prevouts);

    // create a short lived transaction
    let _sighash_tx = psbt.unsigned_tx.clone();
    let mut sighash_cache = SighashCache::new(&_sighash_tx);

    match required_sig {
        RequiredSig::Ecdsa(_) => todo!(),
        RequiredSig::SchnorrTapKey(req_key) => {
            let schnorr_sighashty = psbt.inputs[input_index]
                .sighash_type
                .map(|ty| {
                    ty.schnorr_hash_ty()
                        .expect("Invalid sighash for schnorr sig")
                })
                .unwrap_or(SchnorrSighashType::Default);
            let sighash = sighash_cache.taproot_key_spend_signature_hash(
                input_index,
                &prevouts,
                schnorr_sighashty,
            )?;

            // TODO: ideally here we would look at the PSBT key origins to get the right key and
            // figure out how to derive it. I'm lazy so I'll just iterate over all of them and
            // see if they match

            let (deriv_path, secret_key) = match keymap
                .iter()
                .find_map(|(pk, sk)| pk.is_parent(req_key).map(|path| (path, sk)))
            {
                Some(v) => v,
                None => return Ok(false),
            };
            let secret_key = match secret_key {
                DescriptorSecretKey::Single(single) => single.key.inner,
                DescriptorSecretKey::XPrv(xprv) => {
                    xprv.xkey
                        .derive_priv(&secp, &xprv.derivation_path.extend(deriv_path))?
                        .private_key
                }
            };

            let pubkey = PublicKey::from_secret_key(&secp, &secret_key);
            let x_only_pubkey = XOnlyPublicKey::from(pubkey);

            let tweak = taproot::TapTweakHash::from_key_and_tweak(
                x_only_pubkey,
                psbt.inputs[input_index].tap_merkle_root.clone(),
            );
            let keypair = KeyPair::from_secret_key(&secp, &secret_key.clone())
                .add_xonly_tweak(&secp, &tweak.to_scalar())
                .unwrap();

            let msg = Message::from_slice(sighash.as_ref()).expect("Sighashes are 32 bytes");
            let sig = secp.sign_schnorr_no_aux_rand(&msg, &keypair);

            let bitcoin_sig = SchnorrSig {
                sig,
                hash_ty: schnorr_sighashty,
            };

            psbt.inputs[input_index].tap_key_sig = Some(bitcoin_sig);
            Ok(true)
        }
        RequiredSig::SchnorrTapScript(req_key, leaf_hash) => {
            let sighash_type = psbt.inputs[input_index]
                .sighash_type
                .map(|ty| {
                    ty.schnorr_hash_ty()
                        .expect("Invalid sighash for schnorr sig")
                })
                .unwrap_or(SchnorrSighashType::Default);
            let sighash = sighash_cache.taproot_script_spend_signature_hash(
                input_index,
                &prevouts,
                *leaf_hash,
                sighash_type,
            )?;

            let (deriv_path, secret_key) = match keymap
                .iter()
                .find_map(|(pk, sk)| pk.is_parent(req_key).map(|path| (path, sk)))
            {
                Some(v) => v,
                None => return Ok(false),
            };
            let secret_key = match secret_key {
                DescriptorSecretKey::Single(single) => single.key.inner,
                DescriptorSecretKey::XPrv(xprv) => {
                    xprv.xkey
                        .derive_priv(&secp, &xprv.derivation_path.extend(deriv_path))?
                        .private_key
                }
            };
            let keypair = KeyPair::from_secret_key(&secp, &secret_key.clone());
            let x_only_pubkey = XOnlyPublicKey::from(keypair.public_key());
            let msg = Message::from_slice(sighash.as_ref()).expect("Sighashes are 32 bytes");
            let sig = secp.sign_schnorr_no_aux_rand(&msg, &keypair);
            let bitcoin_sig = SchnorrSig {
                sig,
                hash_ty: sighash_type,
            };

            psbt.inputs[input_index]
                .tap_script_sigs
                .insert((x_only_pubkey, *leaf_hash), bitcoin_sig);
            Ok(true)
        }
    }
}
