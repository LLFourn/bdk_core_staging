use bdk_chain::ConfirmationTime;
use bdk_cli::{
    anyhow::{Ok, Result},
    Broadcast,
};
use bdk_esplora_example::{
    bitcoin::{Amount, Network},
    esplora::Client,
    DEFAULT_PARALLEL_REQUESTS,
};
use electrsd::{
    bitcoind::{
        self,
        bitcoincore_rpc::{bitcoincore_rpc_json::AddressType, RpcApi},
        BitcoinD,
    },
    electrum_client::ElectrumApi,
    ElectrsD,
};
use std::env;
use std::time::{Duration, Instant};

fn setup_electrs_server(bitcoind: &BitcoinD) -> ElectrsD {
    let electrs_exe = env::var("ELECTRS_EXE")
        .ok()
        .or_else(electrsd::downloaded_exe_path)
        .expect("you need to provide env var ELECTRS_EXE or specify an electrsd version feature");
    let mut conf = electrsd::Conf::default();
    conf.http_enabled = true;
    ElectrsD::with_conf(electrs_exe, bitcoind, &conf).unwrap()
}

fn setup_bitcoind(conf: Option<bitcoind::Conf>) -> BitcoinD {
    let bitcoind_exe = env::var("BITCOIND_EXE")
        .ok()
        .or_else(|| bitcoind::downloaded_exe_path().ok())
        .expect(
            "you need to provide an env var BITCOIND_EXE or specify a bitcoind version feature",
        );
    let conf = conf.unwrap_or_else(|| bitcoind::Conf::default());

    BitcoinD::with_conf(bitcoind_exe, &conf).unwrap()
}

fn wait_for_tx_appears_in_esplora(
    wait_seconds: u64,
    electrs_daemon: &ElectrsD,
    txid: &bdk_chain::bitcoin::Txid,
) -> bool {
    let instant = Instant::now();
    loop {
        let wait_tx = electrs_daemon.client.transaction_get(txid);
        if wait_tx.is_ok() {
            return true;
        }

        if instant.elapsed() >= Duration::from_secs(wait_seconds) {
            return false;
        }
    }
}

fn generate_blocks_and_wait(num: usize, bitcoin_daemon: &BitcoinD, electrs_daemon: &ElectrsD) {
    let curr_height = bitcoin_daemon.client.get_block_count().unwrap();
    generate_blocks(num, bitcoin_daemon);
    wait_for_block(curr_height as usize + num, electrs_daemon);
}

fn reorg(num_blocks: usize, bitcoin_daemon: &BitcoinD) -> Result<()> {
    let best_hash = bitcoin_daemon.client.get_best_block_hash()?;
    let initial_height = bitcoin_daemon.client.get_block_info(&best_hash)?.height;

    let mut to_invalidate = best_hash;
    for i in 1..=num_blocks {
        println!(
            "Invalidating block {}/{} ({})",
            i, num_blocks, to_invalidate
        );

        bitcoin_daemon.client.invalidate_block(&to_invalidate)?;
        to_invalidate = bitcoin_daemon.client.get_best_block_hash()?;
    }

    println!(
        "Invalidated {} blocks to new height of {}",
        num_blocks,
        initial_height - num_blocks as usize
    );

    Ok(())
}

fn premine(num_blocks: usize, bitcoin_daemon: &BitcoinD, electrs_daemon: &ElectrsD) {
    generate_blocks_and_wait(num_blocks, bitcoin_daemon, electrs_daemon);
}

fn setup_esplora_client(electrs_daemon: &ElectrsD) -> Client {
    let esplora_url = format!("http://{}", electrs_daemon.esplora_url.as_ref().unwrap());
    Client::new(&esplora_url, DEFAULT_PARALLEL_REQUESTS)
        .expect("creation of Rust Esplora Client failed")
}

fn generate_blocks(num: usize, bitcoin_daemon: &BitcoinD) {
    let address = bitcoin_daemon
        .client
        .get_new_address(Some("test"), Some(AddressType::Bech32))
        .unwrap();
    let _block_hashes = bitcoin_daemon
        .client
        .generate_to_address(num as u64, &address)
        .unwrap();
}

fn wait_for_block(min_height: usize, electrs_daemon: &ElectrsD) {
    let mut header = electrs_daemon.client.block_headers_subscribe().unwrap();
    loop {
        if header.height >= min_height {
            break;
        }
        header = exponential_backoff_poll(|| {
            electrs_daemon.trigger().unwrap();
            electrs_daemon.client.ping().unwrap();
            electrs_daemon.client.block_headers_pop().unwrap()
        });
    }
}

fn exponential_backoff_poll<T, F>(mut poll: F) -> T
where
    F: FnMut() -> Option<T>,
{
    let mut delay = Duration::from_millis(64);
    loop {
        match poll() {
            Some(data) => break data,
            None if delay.as_millis() < 512 => delay = delay.mul_f32(2.0),
            None => {}
        }

        std::thread::sleep(delay);
    }
}

#[test]
fn test_confirmed_balance() -> Result<()> {
    // setup bitcoin and electrs daemon
    let bitcoin_daemon = setup_bitcoind(None);
    let electrs_daemon = setup_electrs_server(&bitcoin_daemon);

    premine(101, &bitcoin_daemon, &electrs_daemon);

    let client = setup_esplora_client(&electrs_daemon);

    // initialize the bdk_esplora_example for testing
    let (_keymap, keychain_tracker, db) = bdk_esplora_example::init()?;

    let address = bdk_esplora_example::get_new_address(&keychain_tracker, &db, Network::Regtest)?;

    // Send bitcoins to the above address
    let _txid = bitcoin_daemon
        .client
        .send_to_address(
            &address,
            Amount::from_sat(1000),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // mine a block to confirm the transaction
    generate_blocks_and_wait(1, &bitcoin_daemon, &electrs_daemon);

    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);
    assert_eq!(balance.confirmed, 1000);

    Ok(())
}

#[test]
fn test_unconfirmed_balance() -> Result<()> {
    // setup bitcoin and electrs daemons
    let bitcoin_daemon = setup_bitcoind(None);
    let electrs_daemon = setup_electrs_server(&bitcoin_daemon);

    premine(101, &bitcoin_daemon, &electrs_daemon);

    let client = setup_esplora_client(&electrs_daemon);

    // initialize the bdk_esplora_example for testing
    let (_keymap, keychain_tracker, db) = bdk_esplora_example::init()?;

    let address = bdk_esplora_example::get_new_address(&keychain_tracker, &db, Network::Regtest)?;

    // Send bitcoins to the above address
    let txid = bitcoin_daemon
        .client
        .send_to_address(
            &address,
            Amount::from_sat(1000),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    wait_for_tx_appears_in_esplora(5, &electrs_daemon, &txid);

    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);
    assert_eq!(balance.untrusted_pending, 1000);

    Ok(())
}

#[test]
fn test_reorg() -> Result<()> {
    //setup bitcoin nodes and electrsd
    let mut conf = bitcoind::Conf::default();
    conf.p2p = bitcoind::P2P::Yes;
    let bitcoin_daemon = setup_bitcoind(Some(conf));
    let electrs_daemon = setup_electrs_server(&bitcoin_daemon);
    let mut miner_conf = bitcoind::Conf::default();
    miner_conf.p2p = bitcoin_daemon.p2p_connect(true).unwrap();
    let miner_node = setup_bitcoind(Some(miner_conf));

    premine(101, &bitcoin_daemon, &electrs_daemon);

    let client = setup_esplora_client(&electrs_daemon);

    // initialize the bdk_esplora_example app
    let (_keymap, keychain_tracker, db) = bdk_esplora_example::init()?;

    //generate an address and send to that address.
    let address = bdk_esplora_example::get_new_address(&keychain_tracker, &db, Network::Regtest)?;

    //Send bitcoins to the address above
    let txid = bitcoin_daemon
        .client
        .send_to_address(
            &address,
            Amount::from_sat(1000),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    //Mine a couple of blocks
    generate_blocks_and_wait(1, &bitcoin_daemon, &electrs_daemon);

    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    //check balance of wallet
    let balance = keychain_tracker.lock().unwrap().balance(|_| false);

    assert_eq!(balance.confirmed, 1000);

    // Reorg blocks on miner chain
    reorg(3, &miner_node)?;
    // Generate more blocks on the miner node, thereby making it the chain with the most
    // work, so the bitcoin_daemon chain has to catch up on this chain which doesn't
    // have a transaction above.
    generate_blocks_and_wait(5, &miner_node, &electrs_daemon);

    //test if you find your transaction in pending untrusted balance.
    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);
    assert_eq!(balance.untrusted_pending, 1000);

    let (new_pos, tx) = bdk_esplora_example::get_tx_and_confirmation(txid, &keychain_tracker);
    assert_eq!(new_pos, ConfirmationTime::Unconfirmed);

    bitcoin_daemon.client.send_raw_transaction(&tx).unwrap();
    generate_blocks_and_wait(1, &bitcoin_daemon, &electrs_daemon);

    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);
    assert_eq!(balance.confirmed, 1000);

    Ok(())
}

#[test]
fn test_send_tx() -> Result<()> {
    let bitcoin_daemon = setup_bitcoind(None);
    let electrs_daemon = setup_electrs_server(&bitcoin_daemon);

    let client = setup_esplora_client(&electrs_daemon);

    premine(101, &bitcoin_daemon, &electrs_daemon);

    // Initialize the bdk esplora app
    let (keymap, keychain_tracker, db) = bdk_esplora_example::init()?;

    let address = bdk_esplora_example::get_new_address(&keychain_tracker, &db, Network::Regtest)?;

    // Send bitcoins to the address above
    let _ = bitcoin_daemon
        .client
        .send_to_address(
            &address,
            Amount::from_sat(10000),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

    // Get the transaction confirmed above
    generate_blocks_and_wait(1, &bitcoin_daemon, &electrs_daemon);

    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);

    assert_eq!(balance.confirmed, 10000);

    let address = bitcoin_daemon.client.get_new_address(None, None)?;

    // Create a transaction that sends back coins to bitcoind
    let (tx, _) = bdk_cli::create_tx(
        1000,
        address,
        bdk_cli::CoinSelectionAlgo::BranchAndBound,
        &mut keychain_tracker.lock().unwrap(),
        &keymap,
    )?;

    let _ = client.broadcast(&tx)?;

    // Get transaction confirmed
    generate_blocks_and_wait(1, &bitcoin_daemon, &electrs_daemon);

    // Sync esplora client
    bdk_esplora_example::run_sync(false, false, true, &keychain_tracker, &db, &client)?;

    let balance = keychain_tracker.lock().unwrap().balance(|_| false);

    assert_eq!(balance.confirmed, 8716);
    let tracker_lock = keychain_tracker.lock().unwrap();
    let (_, chain_tx) = tracker_lock
        .chain_graph()
        .get_tx_in_chain(tx.txid())
        .expect("transaction should be still in chain");
    assert_eq!(chain_tx.clone(), tx);

    Ok(())
}
