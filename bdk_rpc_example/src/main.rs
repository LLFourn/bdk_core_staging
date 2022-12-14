mod rpc;

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
        Arc,
    },
    time::{Duration, SystemTime},
};

use bdk_chain::keychain::KeychainChangeSet;
use bdk_chain::{chain_graph::ChainGraph, BlockId, TxHeight};
use bdk_cli::{
    anyhow,
    clap::{self, Args, Subcommand},
};
use bitcoincore_rpc::Auth;
use rpc::{Client, RpcData, RpcError};

const CHANNEL_BOUND: usize = 10;
const LIVE_POLL_DUR_SECS: u64 = 15;
const FALLBACK_CP_LIMIT: usize = 100;

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
        /// The unused-scripts lookahead will be kept at this size
        #[clap(long, default_value = "10")]
        lookahead: u32,
        /// Whether to be live!
        #[clap(long, default_value = "false")]
        live: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let sigterm_flag = start_ctrlc_handler();

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
            lookahead,
            live,
        } => {
            let (chan, recv) = sync_channel::<RpcData>(CHANNEL_BOUND);
            let mut local_cps = keychain_tracker.chain().checkpoints().clone();

            // emit blocks thread
            let thread_flag = sigterm_flag.clone();
            let cp_limit = keychain_tracker
                .checkpoint_limit()
                .unwrap_or(FALLBACK_CP_LIMIT);
            let join_handle = std::thread::spawn(move || loop {
                client.emit_blocks(&chan, &mut local_cps, cp_limit, fallback_height)?;
                if live && !await_flag(&thread_flag, Duration::from_secs(LIVE_POLL_DUR_SECS)) {
                    continue;
                }
                return Ok::<_, RpcError>(());
            });

            let mut tip = 0;
            let mut start_time = None;

            for data in recv.iter() {
                if sigterm_flag.load(Ordering::Relaxed) {
                    println!("terminating...");
                    return Ok(());
                }

                let mut update = ChainGraph::<TxHeight>::default();

                let txs = match data {
                    RpcData::Start {
                        local_tip,
                        target_tip,
                    } => {
                        let now = SystemTime::now();

                        println!(
                            "sync start: time={:?}, current_tip={}, target_tip={}",
                            now, local_tip, target_tip
                        );

                        tip = target_tip;
                        start_time = Some(SystemTime::now());
                        continue;
                    }
                    RpcData::Synced => {
                        let balance = keychain_tracker
                            .full_utxos()
                            .map(|(_, utxo)| utxo.txout.value)
                            .sum::<u64>();
                        let duration = start_time
                            .map(|t| t.elapsed().expect("should succeed").as_secs())
                            .unwrap_or(0);
                        println!(
                            "sync finished: duration={}s, tip={}, balance={}sats",
                            duration, tip, balance
                        );
                        continue;
                    }
                    RpcData::Blocks { last_cp, blocks } => {
                        let checkpoints = blocks
                            .iter()
                            .map(|(h, b)| (*h, b.block_hash()))
                            .chain(last_cp);
                        for (height, hash) in checkpoints {
                            let _ = update.insert_checkpoint(BlockId { height, hash })?;
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
                    keychain_tracker.txout_index.pad_all_with_unused(lookahead);
                    if keychain_tracker.txout_index.is_relevant(&tx) {
                        println!("* adding tx to update: {} @ {}", tx.txid(), height);
                        let _ = update.insert_tx(tx.clone(), height)?;
                    }
                    keychain_tracker.txout_index.scan(&tx);
                }

                // TODO: We either need to prune here, or maintain 2 "indexes" in `KeychainTxOut`.
                // 2 indexes meaning one for user-derived scripts, one for maintaining the stop gap.

                let new_indexes = keychain_tracker.txout_index.last_active_indicies();

                let changeset = KeychainChangeSet {
                    derivation_indices: keychain_tracker
                        .txout_index
                        .keychains()
                        .keys()
                        .filter_map(|keychain| {
                            let old_index = old_indexes.get(keychain);
                            let new_index = new_indexes.get(keychain);

                            match new_index {
                                Some(new_ind) if new_index > old_index => {
                                    Some((keychain.clone(), *new_ind))
                                }
                                _ => None,
                            }
                        })
                        .collect(),
                    chain_graph: keychain_tracker
                        .chain_graph()
                        .determine_changeset(&update)?,
                };

                println!(
                    "* scanned_to: {} / {} tip",
                    match changeset.chain_graph.chain.checkpoints.iter().next_back() {
                        Some((height, _)) => height.to_string(),
                        None => "mempool".to_string(),
                    },
                    tip,
                );

                db.append_changeset(&changeset)?;
                keychain_tracker.apply_changeset(changeset);
            }

            join_handle
                .join()
                .expect("failed to join emit_blocks thread")?;
        }
    }

    Ok(())
}

fn start_ctrlc_handler() -> Arc<AtomicBool> {
    let flag = Arc::new(AtomicBool::new(false));
    let cloned_flag = flag.clone();

    ctrlc::set_handler(move || cloned_flag.store(true, Ordering::SeqCst))
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
