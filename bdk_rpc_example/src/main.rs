mod rpc;

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{sync_channel, SyncSender},
        Arc,
    },
    time::{Duration, SystemTime},
};

use bdk_cli::{
    anyhow,
    clap::{self, Args, Subcommand},
    log,
};
use bdk_chain::{chain_graph::ChainGraph, BlockId, TxHeight};
use bdk_chain::keychain::KeychainScan;
use bitcoincore_rpc::Auth;
use rpc::{Client, RpcData, RpcError};

const CHANNEL_BOUND: usize = 1000;

#[derive(Args, Debug, Clone)]
struct RpcArgs {
    /// RPC URL
    #[clap(env = "RPC_URL", long, default_value = "127.0.0.1:8332")]
    url: String,
    /// RPC auth cookie file
    #[clap(env = "RPC_COOKIE", long)]
    rpc_cookie: Option<PathBuf>,
    /// RPC auth username
    #[clap(env = "RPC_USER", long)]
    rpc_user: Option<String>,
    /// RPC auth password
    #[clap(env = "RPC_PASS", long)]
    rpc_password: Option<String>,
}

impl From<RpcArgs> for Auth {
    fn from(args: RpcArgs) -> Self {
        match (args.rpc_cookie, args.rpc_user, args.rpc_password) {
            (None, None, None) => Self::None,
            (Some(path), _, _) => Self::CookieFile(path),
            (_, Some(user), Some(pass)) => Self::UserPass(user, pass),
            (_, Some(_), None) => panic!("rpc auth: missing rpc_pass"),
            (_, None, Some(_)) => panic!("rpc auth: missing rpc_user"),
        }
    }
}

#[derive(Subcommand, Debug, Clone)]
enum RpcCommands {
    /// Scans blocks via RPC (starting from last point of agreement) and stores/indexes relevant
    /// transactions
    Scan {
        /// Starting block height to fallback to if no point of agreement if found
        #[clap(long, default_value = "0")]
        fallback_height: u32,
        /// The unused-script gap will be kept at this value
        #[clap(long, default_value = "10")]
        stop_gap: u32,
        /// Whether to be live!
        #[clap(long, default_value = "false")]
        live: bool,
    },
}

fn main() -> anyhow::Result<()> {
    println!("Loading wallet from db...");
    let (args, keymap, mut keychain_tracker, mut db) =
        bdk_cli::init::<RpcArgs, RpcCommands, TxHeight>()?;
    println!("Wallet loaded.");

    let client = {
        let rpc_url = args.chain_args.url.clone();
        let rpc_auth = args.chain_args.into();
        Client::new(&rpc_url, rpc_auth)?
    };

    let rpc_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(rpc_cmd) => rpc_cmd,
        general_cmd => {
            return bdk_cli::handle_commands(
                general_cmd,
                client,
                &mut keychain_tracker,
                &mut db,
                args.network,
                &keymap,
            )
        }
    };

    match rpc_cmd {
        RpcCommands::Scan {
            fallback_height,
            stop_gap,
            live,
        } => {
            let (chan, recv) = sync_channel::<RpcData>(CHANNEL_BOUND);
            let sigterm_flag = start_ctrlc_handler(chan.clone());
            let local_cps = keychain_tracker.chain().checkpoints().clone();

            // emit blocks thread
            let join_handle = std::thread::spawn(move || loop {
                client.emit_blocks(&chan, &local_cps, fallback_height)?;
                if live && !await_flag(&sigterm_flag, Duration::from_secs(10)) {
                    continue;
                }
                return chan.send(RpcData::Stop(true)).map_err(RpcError::Send);
            });

            let mut tip = 0;

            for data in recv.iter() {
                let mut update = ChainGraph::<TxHeight>::default();
                let is_synced = matches!(&data, &RpcData::Mempool(_));

                let txs = match data {
                    RpcData::Start {
                        local_tip,
                        target_tip,
                    } => {
                        tip = target_tip;
                        println!(
                            "sync start: current_tip={}, target_tip={}",
                            local_tip, target_tip
                        );
                        continue;
                    }
                    RpcData::Stop(finished) => {
                        println!("terminating... sync_finished={}", finished);
                        drop(recv);
                        break;
                    }
                    RpcData::Blocks { last_cp, blocks } => {
                        let checkpoints = blocks
                            .iter()
                            .map(|(h, b)| (*h, b.block_hash()))
                            .chain(last_cp);
                        for (height, hash) in checkpoints {
                            update.insert_checkpoint(BlockId { height, hash })?;
                        }

                        blocks
                            .into_iter()
                            .flat_map(|(height, block)| {
                                block
                                    .txdata
                                    .into_iter()
                                    .map(move |tx| (TxHeight::Confirmed(height), tx))
                            })
                            .collect::<Vec<_>>()
                    }
                    RpcData::Mempool(txs) => txs
                        .into_iter()
                        .map(|tx| (TxHeight::Unconfirmed, tx))
                        .collect::<Vec<_>>(),
                };

                let old_indexes = keychain_tracker.txout_index.derivation_indices();

                for (height, tx) in txs {
                    keychain_tracker
                        .txout_index
                        .derive_until_unused_gap(stop_gap);
                    if keychain_tracker.txout_index.is_relevant(&tx) {
                        println!("* adding tx to update: {} @ {}", tx.txid(), height);
                        update.insert_tx(tx.clone(), height)?;
                    }
                    keychain_tracker.txout_index.scan(&tx);
                }

                keychain_tracker
                    .txout_index
                    .prune_unused(old_indexes.clone());
                let new_indexes = keychain_tracker.txout_index.last_active_indicies();

                let wallet_scan = KeychainScan {
                    update,
                    last_active_indexes: new_indexes.clone(),
                };

                let mut changeset = keychain_tracker.determine_changeset(&wallet_scan)?;
                changeset.derivation_indices = keychain_tracker
                    .txout_index
                    .keychains()
                    .keys()
                    .filter_map(|keychain| {
                        match (old_indexes.get(keychain), new_indexes.get(keychain)) {
                            (Some(old_index), Some(new_index)) if new_index > old_index => {
                                Some((keychain.clone(), *new_index))
                            }
                            (None, Some(new_index)) => Some((keychain.clone(), *new_index)),
                            _ => None,
                        }
                    })
                    .collect();

                db.append_changeset(&changeset)?;

                println!("* index_changes: {:?}", changeset.derivation_indices);
                println!(
                    "* tx_changes  : {}",
                    changeset.chain_graph.chain.txids.len()
                );
                println!(
                    "* scanned: {} / {} tip",
                    match changeset.chain_graph.chain.checkpoints.iter().next_back() {
                        Some((height, _)) => height.to_string(),
                        None => "mempool".to_string(),
                    },
                    tip,
                );
                keychain_tracker.apply_changeset(changeset);

                println!("...");

                if is_synced {
                    let balance = keychain_tracker
                        .full_utxos()
                        .map(|(_, utxo)| utxo.txout.value)
                        .sum::<u64>();
                    println!("sync complete: balance={}", balance);
                    // TODO: Print more stuff
                }
            }

            join_handle
                .join()
                .expect("failed to join emit_blocks thread")?;
        }
    }

    Ok(())
}

fn start_ctrlc_handler(chan: SyncSender<RpcData>) -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let cloned_flag = flag.clone();

    ctrlc::set_handler(move || {
        cloned_flag.store(true, Ordering::SeqCst);
        match chan.send(RpcData::Stop(false)) {
            Ok(_) => log::info!("caught SIGTERM"),
            Err(err) => log::warn!("failed to send SIGTERM: {}", err),
        }
    })
    .expect("failed to set Ctrl+C handler");

    flag
}

fn await_flag(flag: &AtomicBool, duration: Duration) -> bool {
    let start = SystemTime::now();
    loop {
        if flag.load(Ordering::Relaxed) {
            return true;
        }
        if SystemTime::now()
            .duration_since(start)
            .expect("should succeed")
            >= duration
        {
            return false;
        }
        std::thread::sleep(Duration::from_secs(1));
    }
}
