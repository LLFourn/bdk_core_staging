//! This crate is used for updating structures of [`bdk_chain`] with data from an esplora server.
//!
//! The star of the show is the  [`EsploraExt::scan`] method which scans for relevant
//! blockchain data (via esplora) and outputs a [`KeychainScan`].

use bdk_chain::{
    bitcoin::{BlockHash, OutPoint, Script, Transaction, Txid},
    chain_graph::ChainGraph,
    keychain::KeychainScan,
    BlockId, ConfirmationTime,
};
use esplora_client::{OutputStatus, TxStatus};
use std::{
    collections::BTreeMap,
    sync::{Arc, Condvar, Mutex},
};

pub use esplora_client;
use esplora_client::Error;

/// Trait to extend [`esplora_client::BlockingClient`] functionality.
///
/// Refer to [crate-level documentation] for more.
///
/// [crate-level documentation]: crate
pub trait EsploraExt {
    /// Scan the blockchain (via esplora) for the data specified and returns a [`KeychainScan`].
    ///
    /// - `local_chain`: the most recent block hashes present locally
    /// - `keychain_spks`: keychains that we want to scan transactions for
    /// - `txids`: transactions that we want updated [`ChainPosition`]s for
    /// - `outpoints`: transactions associated with these outpoints (residing, spending) that we
    ///     want to included in the update
    ///
    /// The scan for each keychain stops after a gap of `stop_gap` script pubkeys with no associated
    /// transactions. `parallel_requests` specifies the max task-threads allowed.
    ///
    /// [`ChainPosition`]: bdk_chain::sparse_chain::ChainPosition
    fn scan<K: Ord + Clone>(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
        keychain_spks: BTreeMap<K, impl IntoIterator<Item = (u32, Script)>>,
        txids: impl IntoIterator<Item = Txid>,
        outpoints: impl IntoIterator<Item = OutPoint>,
        stop_gap: usize,
        parallel_requests: core::num::NonZeroU8,
    ) -> Result<KeychainScan<K, ConfirmationTime>, Error>;

    /// Convenience method to call [`scan`] without requiring a keychain.
    ///
    /// [`scan`]: EsploraExt::scan
    fn scan_without_keychain(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
        misc_spks: impl Iterator<Item = Script>,
        txids: impl IntoIterator<Item = Txid>,
        outpoints: impl IntoIterator<Item = OutPoint>,
        parallel_requests: core::num::NonZeroU8,
    ) -> Result<ChainGraph<ConfirmationTime>, Error> {
        let wallet_scan = self.scan(
            local_chain,
            [((), misc_spks.enumerate().map(|(i, spk)| (i as u32, spk)))].into(),
            txids,
            outpoints,
            usize::MAX,
            parallel_requests,
        )?;

        Ok(wallet_scan.update)
    }
}

impl EsploraExt for esplora_client::BlockingClient {
    fn scan<K: Ord + Clone>(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
        keychain_spks: BTreeMap<K, impl IntoIterator<Item = (u32, Script)>>,
        txids: impl IntoIterator<Item = Txid>,
        outpoints: impl IntoIterator<Item = OutPoint>,
        stop_gap: usize,
        parallel_requests: core::num::NonZeroU8,
    ) -> Result<KeychainScan<K, ConfirmationTime>, Error> {
        let mut update = prepare_update(self, local_chain)?;

        let tip = update
            .chain()
            .latest_checkpoint()
            .expect("chain must have a checkpoint");

        let mut last_active_indices = BTreeMap::new();

        for (keychain, spks) in keychain_spks {
            let mut spks = spks.into_iter();
            let mut last_active_index = None;
            let mut empty_scripts = 0;

            loop {
                let handles = (0..parallel_requests.get())
                    .filter_map(
                        |_| -> Option<
                            std::thread::JoinHandle<Result<(u32, Vec<esplora_client::Tx>), _>>,
                        > {
                            let (index, script) = spks.next()?;
                            let client = self.clone();
                            Some(std::thread::spawn(move || {
                                let mut related_txs = client.scripthash_txs(&script, None)?;

                                let n_confirmed =
                                    related_txs.iter().filter(|tx| tx.status.confirmed).count();
                                // esplora pages on 25 confirmed transactions. If there's 25 or more we
                                // keep requesting to see if there's more.
                                if n_confirmed >= 25 {
                                    loop {
                                        let new_related_txs = client.scripthash_txs(
                                            &script,
                                            Some(related_txs.last().unwrap().txid),
                                        )?;
                                        let n = new_related_txs.len();
                                        related_txs.extend(new_related_txs);
                                        // we've reached the end
                                        if n < 25 {
                                            break;
                                        }
                                    }
                                }

                                Result::<_, esplora_client::Error>::Ok((index, related_txs))
                            }))
                        },
                    )
                    .collect::<Vec<_>>();

                let n_handles = handles.len();

                for handle in handles {
                    let (index, related_txs) = handle.join().unwrap()?; // TODO: don't unwrap
                    if related_txs.is_empty() {
                        empty_scripts += 1;
                    } else {
                        last_active_index = Some(index);
                        empty_scripts = 0;
                    }
                    for tx in related_txs {
                        let confirmation_time = if tx.status.confirmed
                            // anything higher means that chain tip has progressed
                            && tx.status.block_height.expect("height expected") <= tip.height as _
                        {
                            ConfirmationTime::Confirmed {
                                height: tx.status.block_height.expect("height expected"),
                                time: tx.status.block_time.expect("blocktime expected"),
                            }
                        } else {
                            ConfirmationTime::Unconfirmed
                        };
                        if let Err(failure) = update.insert_tx(tx.to_tx(), confirmation_time) {
                            use bdk_chain::{
                                chain_graph::InsertTxError, sparse_chain::InsertTxError::*,
                            };
                            match failure {
                                InsertTxError::Chain(TxTooHigh { .. }) => {
                                    unreachable!("chain position already checked earlier")
                                }
                                InsertTxError::Chain(TxMovedUnexpectedly { .. }) => {
                                    /* Reorg occured (catch error below), ignore tx for now */
                                }
                                InsertTxError::UnresolvableConflict(_) => {
                                    /* Reorg occured (catch error below), ignore tx for now */
                                }
                            }
                        }
                    }
                }

                if n_handles == 0 || empty_scripts >= stop_gap {
                    break;
                }
            }

            if let Some(last_active_index) = last_active_index {
                last_active_indices.insert(keychain, last_active_index);
            }
        }

        let mut txid_tasks =
            TaskSpawn::<Txid, Option<(Transaction, TxStatus)>, Error>::new(parallel_requests);
        for txid in txids.into_iter() {
            let client = self.clone();
            if !txid_tasks.wait_and_spawn(txid, move || {
                let tx_status = match client.get_tx_status(&txid)? {
                    Some(status) => status,
                    None => return Ok(None),
                };
                let tx = match client.get_tx(&txid)? {
                    Some(tx) => tx,
                    None => return Ok(None),
                };
                Ok(Some((tx, tx_status)))
            }) {
                break;
            }
        }
        for (_txid, result) in txid_tasks.join_all() {
            let (tx, status): (Transaction, TxStatus) = match result? {
                Some(v) => v,
                None => continue,
            };

            let confirmation_time = if status.confirmed
                && status.block_height.expect("height expected") <= tip.height as _
            {
                ConfirmationTime::Confirmed {
                    height: status.block_height.expect("height expected"),
                    time: status.block_time.expect("blocktime expected"),
                }
            } else {
                ConfirmationTime::Unconfirmed
            };

            if let Err(failure) = update.insert_tx(tx, confirmation_time) {
                use bdk_chain::{chain_graph::InsertTxError, sparse_chain::InsertTxError::*};
                match failure {
                    InsertTxError::Chain(TxTooHigh { .. }) => {
                        unreachable!("chain position already checked earlier")
                    }
                    InsertTxError::Chain(TxMovedUnexpectedly { .. }) => {
                        /* Reorg occured (catch error below), ignore tx for now */
                    }
                    InsertTxError::UnresolvableConflict(_) => {
                        /* Reorg occured (catch error below), ignore tx for now */
                    }
                }
            }
        }

        let mut op_tasks =
            TaskSpawn::<OutPoint, Vec<(Transaction, TxStatus)>, Error>::new(parallel_requests);
        for op in outpoints.into_iter() {
            let client = self.clone();
            if !op_tasks.wait_and_spawn(op, move || {
                let mut out = Vec::with_capacity(2);
                match (client.get_tx(&op.txid)?, client.get_tx_status(&op.txid)?) {
                    (Some(tx), Some(tx_status)) => out.push((tx, tx_status)),
                    _ => return Ok(out),
                }
                match client.get_output_status(&op.txid, op.vout as _)? {
                    Some(OutputStatus {
                        txid: Some(txid),
                        status: Some(spend_status),
                        ..
                    }) => {
                        if let Some(spend_tx) = client.get_tx(&txid)? {
                            out.push((spend_tx, spend_status));
                        }
                    }
                    _ => return Ok(out),
                }
                Ok(out)
            }) {
                break;
            }
        }
        for (_op, result) in op_tasks.join_all() {
            for (tx, status) in result? {
                let confirmation_time = if status.confirmed
                    && status.block_height.expect("height expected") <= tip.height as _
                {
                    ConfirmationTime::Confirmed {
                        height: status.block_height.expect("height expected"),
                        time: status.block_time.expect("blocktime expected"),
                    }
                } else {
                    ConfirmationTime::Unconfirmed
                };

                if let Err(failure) = update.insert_tx(tx, confirmation_time) {
                    use bdk_chain::{chain_graph::InsertTxError, sparse_chain::InsertTxError::*};
                    match failure {
                        InsertTxError::Chain(TxTooHigh { .. }) => {
                            unreachable!("chain position already checked earlier")
                        }
                        InsertTxError::Chain(TxMovedUnexpectedly { .. }) => {
                            /* Reorg occured (catch error below), ignore tx for now */
                        }
                        InsertTxError::UnresolvableConflict(_) => {
                            /* Reorg occured (catch error below), ignore tx for now */
                        }
                    }
                }
            }
        }

        // Reorg mitigation logic...
        'mitigate_reorg: loop {
            let recheck_update = prepare_update(self, update.chain().checkpoints())?;
            let changeset = update.apply_update(recheck_update).expect("should work");
            let displaced_txids = changeset.chain.txids.into_keys();

            if displaced_txids.len() == 0 {
                break 'mitigate_reorg;
            }

            // recheck status of all unconfirmed txs
            let mut tasks = TaskSpawn::<Txid, Option<TxStatus>, Error>::new(parallel_requests);
            for txid in displaced_txids {
                let client = self.clone();
                if !tasks.wait_and_spawn(txid, move || client.get_tx_status(&txid)) {
                    break;
                }
            }
            for (txid, result) in tasks.join_all() {
                let status = match result? {
                    Some(status) => status,
                    None => continue, // [TODO] Should we remove tx from unconfirmed here?
                };

                let confirmation_time = if status.confirmed
                    // anything higher means that chain tip has progressed
                    && status.block_height.expect("height expected") <= tip.height as _
                {
                    ConfirmationTime::Confirmed {
                        height: status.block_height.expect("height expected"),
                        time: status.block_time.expect("blocktime expected"),
                    }
                } else {
                    ConfirmationTime::Unconfirmed
                };

                // [TODO] Should we allow directly updating tx position on chaingraph?
                let tx = update.graph().get_tx(txid).expect("must exist").clone();
                if let Err(failure) = update.insert_tx(tx, confirmation_time) {
                    use bdk_chain::{chain_graph::InsertTxError, sparse_chain::InsertTxError::*};
                    match failure {
                        InsertTxError::Chain(TxTooHigh { .. }) => {
                            unreachable!("chain position already checked earlier")
                        }
                        InsertTxError::Chain(TxMovedUnexpectedly { .. }) => {
                            /* Reorg occured (catch error below), ignore tx for now */
                        }
                        InsertTxError::UnresolvableConflict(_) => {
                            /* Reorg occured (catch error below), ignore tx for now */
                        }
                    }
                }
            }
        }

        Ok(KeychainScan {
            update,
            last_active_indices,
        })
    }
}

struct TaskSpawn<ID, T, E> {
    max_tasks: usize,
    join_handles: Vec<std::thread::JoinHandle<(ID, Result<T, E>)>>,
    ctrl: Arc<(Mutex<usize>, Condvar)>,
    err_flag: Arc<Mutex<bool>>,
}

impl<ID: Send + 'static, T: Send + 'static, E: Send + 'static> TaskSpawn<ID, T, E> {
    fn new(max_tasks: core::num::NonZeroU8) -> Self {
        Self {
            max_tasks: max_tasks.get() as usize,
            join_handles: Default::default(),
            ctrl: Default::default(),
            err_flag: Default::default(),
        }
    }

    fn wait_and_spawn<F>(&mut self, id: ID, f: F) -> bool
    where
        F: Fn() -> Result<T, E> + Send + 'static,
    {
        let (count, cvar) = &*self.ctrl;

        let mut count_guard = count.lock().unwrap();
        while *count_guard >= self.max_tasks {
            count_guard = cvar.wait(count_guard).unwrap();
        }

        if *self.err_flag.lock().unwrap() {
            return false;
        }
        *count_guard += 1;

        let ctrl = Arc::clone(&self.ctrl);
        let err_flag = Arc::clone(&self.err_flag);

        self.join_handles.push(std::thread::spawn(move || {
            let result = f();

            if result.is_err() {
                *err_flag.lock().unwrap() = true;
            }

            let (count, cvar) = &*ctrl;

            count
                .lock()
                .unwrap()
                .checked_sub(1)
                .expect("task count must not overflow");
            cvar.notify_one();

            (id, result)
        }));

        false
    }

    fn join_all(&mut self) -> impl Iterator<Item = (ID, Result<T, E>)> {
        let mut join_handes = Vec::new();
        core::mem::swap(&mut join_handes, &mut self.join_handles);
        join_handes
            .into_iter()
            .map(|handle| handle.join().expect("thread paniced"))
    }
}

fn prepare_update(
    client: &esplora_client::BlockingClient,
    local_chain: &BTreeMap<u32, BlockHash>,
) -> Result<ChainGraph<ConfirmationTime>, Error> {
    'outer: loop {
        let mut update = ChainGraph::<ConfirmationTime>::default();

        'find_agreement: for (&height, &original_hash) in local_chain.iter().rev() {
            let update_block_id = BlockId {
                height,
                hash: client.get_block_hash(height)?,
            };
            let _ = update
                .insert_checkpoint(update_block_id)
                .expect("must not collide");
            if update_block_id.hash == original_hash {
                break 'find_agreement;
            }
        }

        let tip_block_id = 'find_tip: loop {
            let hash = client.get_tip_hash()?;
            match client.get_block_status(&hash)?.height {
                Some(height) => break 'find_tip BlockId { hash, height },
                None => continue 'find_tip,
            }
        };

        if update.insert_checkpoint(tip_block_id).is_err() {
            continue 'outer;
        }

        return Ok(update);
    }
}
