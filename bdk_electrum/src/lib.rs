//! This crate is used for updating [`KeychainTracker`] (from the [`bdk_chain`] crate) with data
//! from an electrum server.
//!
//! The star of the show is the [`ElectrumClient::scan`] method, which scans for relevant blockchain
//! data (via electrum). `scan` takes in [`ScanParams`], and outputs a [`ElectrumUpdate`].
//!
//! A [`ElectrumUpdate`] only includes `txid`s and no full transactions. The caller is responsible
//! for obtaining full transactions before applying it to [`KeychainTracker`]. This can be done with
//! these steps:
//!
//! 1. Determine which full transactions are missing from [`KeychainTracker`]. The method
//! [`find_missing_txids`] of [`ElectrumUpdate`] can be used.
//!
//! 2. Obtaining the full transactions. To do this via electrum, the method
//! [`batch_transaction_get`] can be used.
//!
//! Refer to [`bdk_electrum_example`] for a complete example.
//!
//! [`KeychainTracker`]: bdk_chain::keychain::KeychainTracker
//! [`ElectrumClient::scan`]: ElectrumClient::scan
//! [`find_missing_txids`]: ElectrumUpdate::find_missing_txids
//! [`batch_transaction_get`]: ElectrumApi::batch_transaction_get
//! [`bdk_electrum_example`]: https://github.com/LLFourn/bdk_core_staging/tree/master/bdk_electrum_example

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    ops::Deref,
};

use bdk_chain::{
    bitcoin::{BlockHash, OutPoint, Script, Transaction, Txid},
    chain_graph::{self, ChainGraph},
    keychain::KeychainScan,
    sparse_chain::{self, ChainPosition, SparseChain},
    tx_graph::TxGraph,
    AsTransaction, BlockId, ConfirmationTime, TxHeight,
};
pub use electrum_client;
use electrum_client::Client;
pub use electrum_client::{ElectrumApi, Error};

/// Arguments for getting a [`ChainPosition`] implementation from [`ElectrumClient`].
#[non_exhaustive]
pub struct ChainPositionArgs<'c> {
    pub client: &'c ElectrumClient,
    pub txid: Txid,
    pub height: TxHeight,
}

/// The `to_position` argument for [`scan`] or [`scan_without_keychain`] to use [`TxHeight`] as the
/// [`ChainPosition`].
///
/// [`scan`]: ElectrumClient::scan
/// [`scan_without_keychain`]: ElectrumClient::scan_without_keychain
pub fn tx_height_position<'c>(args: ChainPositionArgs<'c>) -> Result<TxHeight, Error> {
    Ok(args.height)
}

/// The `to_position` argument for [`scan`] or [`scan_without_keychain`] to use [`ConfirmationTime`]
/// as the [`ChainPosition`].
///
/// [`scan`]: ElectrumClient::scan
/// [`scan_without_keychain`]: ElectrumClient::scan_without_keychain
pub fn confirmation_time_position<'c>(
    args: ChainPositionArgs<'c>,
) -> Result<ConfirmationTime, Error> {
    Ok(match args.height {
        TxHeight::Confirmed(height) => {
            let time = args.client.block_header(height as _)?.time as u64;
            ConfirmationTime::Confirmed { height, time }
        }
        TxHeight::Unconfirmed => ConfirmationTime::Unconfirmed,
    })
}

/// Structure to get data from electrum to update [`bdk_chain`] structures.
///
/// This uses [`electrum_client::Client`] internally.
pub struct ElectrumClient {
    inner: Client,
}

impl Deref for ElectrumClient {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ElectrumClient {
    fn position_args(&self, txid: Txid, tx_height: TxHeight) -> ChainPositionArgs {
        ChainPositionArgs {
            client: self,
            txid,
            height: tx_height,
        }
    }

    /// Create a new [`ElectrumClient`] that is connected to electrum server at `url` with default
    /// [`electrum_client::Config`].
    pub fn new(url: &str) -> Result<Self, Error> {
        Ok(Self {
            inner: electrum_client::Client::new(url)?,
        })
    }

    /// Create a new [`ElectrumClient`] from the provided [`electrum_client::Config`].
    pub fn from_config(url: &str, config: electrum_client::Config) -> Result<Self, Error> {
        Ok(Self {
            inner: electrum_client::Client::from_config(url, config)?,
        })
    }

    /// Create a new [`ElectrumClient`] from the provided [`electrum_client::Client`].
    pub fn from_client(client: Client) -> Self {
        Self { inner: client }
    }

    /// Fetch the latest block height.
    pub fn get_tip(&self) -> Result<(u32, BlockHash), Error> {
        // TODO: unsubscribe when added to the client, or is there a better call to use here?
        Ok(self
            .inner
            .block_headers_subscribe()
            .map(|data| (data.height as u32, data.header.block_hash()))?)
    }

    /// Prepare an update sparsechain "template" based on the checkpoints of the `local_chain`.
    fn prepare_update<P: ChainPosition>(
        &self,
        to_position: &impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        local_chain: &BTreeMap<u32, BlockHash>,
    ) -> Result<SparseChain<P>, Error> {
        let mut update = SparseChain::<P>::default();

        // Find local chain block that is still there so our update can connect to the local chain.
        for (&existing_height, &existing_hash) in local_chain.iter().rev() {
            // TODO: a batch request may be safer, as a reorg that happens when we are obtaining
            //       `block_header`s will result in inconsistencies
            let current_hash = self
                .inner
                .block_header(existing_height as usize)?
                .block_hash();
            let _ = update
                .insert_checkpoint(BlockId {
                    height: existing_height,
                    hash: current_hash,
                })
                .expect("This never errors because we are working with a fresh chain");

            if current_hash == existing_hash {
                break;
            }
        }

        // Insert the new tip so new transactions will be accepted into the sparse chain.
        let tip = {
            let (height, hash) = self.get_tip()?;
            BlockId { height, hash }
        };
        if let Err(failure) = update.insert_checkpoint(tip) {
            match failure {
                sparse_chain::InsertCheckpointError::HashNotMatching { .. } => {
                    // There has been a re-org before we even begin scanning addresses.
                    // Just recursively call (this should never happen).
                    return self.prepare_update(to_position, local_chain);
                }
            }
        }

        Ok(update)
    }

    /// Populate an update [`SparseChain`] with transactions (and associated block positions) that
    /// contain and spend the given `outpoints`.
    fn populate_with_outpoints<P: ChainPosition>(
        &self,
        to_position: &impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        update: &mut SparseChain<P>,
        outpoints: impl Iterator<Item = OutPoint>,
    ) -> Result<HashMap<Txid, Transaction>, InternalError> {
        let tip = update
            .latest_checkpoint()
            .expect("update must atleast have one checkpoint");

        let mut full_txs = HashMap::new();
        for outpoint in outpoints {
            let txid = outpoint.txid;
            let tx = self.inner.transaction_get(&txid)?;
            debug_assert_eq!(tx.txid(), txid);
            let txout = match tx.output.get(outpoint.vout as usize) {
                Some(txout) => txout,
                None => continue,
            };

            // attempt to find the following transactions (alongside their chain positions), and
            // add to our sparsechain `update`:
            let mut has_residing = false; // tx in which the outpoint resides
            let mut has_spending = false; // tx that spends the outpoint
            for res in self.inner.script_get_history(&txout.script_pubkey)? {
                if has_residing && has_spending {
                    break;
                }

                // skip if we have already added the tx to our `update`
                if full_txs.contains_key(&res.tx_hash) {
                    if res.tx_hash == txid {
                        has_residing = true;
                    } else {
                        has_spending = true;
                    }
                    continue;
                }

                let res_tx = if res.tx_hash == txid {
                    has_residing = true;
                    tx.clone()
                } else {
                    let res_tx = self.inner.transaction_get(&res.tx_hash)?;
                    if !res_tx
                        .input
                        .iter()
                        .any(|txin| txin.previous_output == outpoint)
                    {
                        continue;
                    }
                    has_spending = true;
                    res_tx
                };
                full_txs.insert(res.tx_hash, res_tx);

                let tx_height = match res.height {
                    // Electrum server API specifies that height <= 0 is considered unconfirmed
                    h if h <= 0 => {
                        debug_assert!(
                            h == 0 || h == -1,
                            "unexpected height ({}) from electrum server",
                            h
                        );
                        TxHeight::Unconfirmed
                    }
                    h => {
                        let h = h as u32;
                        if h > tip.height {
                            TxHeight::Unconfirmed
                        } else {
                            TxHeight::Confirmed(h)
                        }
                    }
                };
                let tx_pos = to_position(self.position_args(txid, tx_height))?;

                if let Err(failure) = update.insert_tx(txid, tx_pos) {
                    match failure {
                        sparse_chain::InsertTxError::TxTooHigh { .. } => {
                            unreachable!(
                                "we should never encounter this as we ensured height <= tip"
                            );
                        }
                        sparse_chain::InsertTxError::TxMovedUnexpectedly { .. } => {
                            return Err(InternalError::Reorg);
                        }
                    }
                }
            }
        }
        Ok(full_txs)
    }

    /// Populate an update [`SparseChain`] with transactions (and associated block positions) from
    /// the given `txids`.
    fn populate_with_txids<P: ChainPosition>(
        &self,
        to_position: &impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        update: &mut SparseChain<P>,
        txids: impl Iterator<Item = Txid>,
    ) -> Result<(), InternalError> {
        let tip = update
            .latest_checkpoint()
            .expect("update must have atleast one checkpoint");
        for txid in txids {
            let tx = match self.inner.transaction_get(&txid) {
                Ok(tx) => tx,
                Err(electrum_client::Error::Protocol(_)) => continue,
                Err(other_err) => return Err(other_err.into()),
            };

            let spk = tx
                .output
                .get(0)
                .map(|txo| &txo.script_pubkey)
                .expect("tx must have an output");

            let tx_height = match self
                .inner
                .script_get_history(spk)?
                .into_iter()
                .find(|r| r.tx_hash == txid)
                .map(|r| r.height)
            {
                Some(h) if h > 0 => TxHeight::Confirmed(h as _),
                Some(_) => TxHeight::Unconfirmed,
                None => continue,
            };

            if tx_height.is_confirmed() && tx_height > TxHeight::Confirmed(tip.height) {
                continue;
            }
            let tx_pos = to_position(self.position_args(txid, tx_height))?;
            if let Err(failure) = update.insert_tx(txid, tx_pos) {
                match failure {
                    sparse_chain::InsertTxError::TxTooHigh { .. } => {
                        unreachable!("we should never encounter this as we ensured height <= tip");
                    }
                    sparse_chain::InsertTxError::TxMovedUnexpectedly { .. } => {
                        return Err(InternalError::Reorg);
                    }
                }
            }
        }
        Ok(())
    }

    /// Populate an update [`SparseChain`] with transactions (and associated block positions) from
    /// the transaction history of the provided `spks`.
    fn populate_with_spks<P, K, I, S>(
        &self,
        to_position: &impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        update: &mut SparseChain<P>,
        spks: &mut S,
        stop_gap: usize,
        batch_size: usize,
    ) -> Result<BTreeMap<I, (Script, bool)>, InternalError>
    where
        P: ChainPosition,
        K: Ord + Clone,
        I: Ord + Clone,
        S: Iterator<Item = (I, Script)>,
    {
        let tip = update.latest_checkpoint().map_or(0, |cp| cp.height);
        let mut unused_spk_count = 0_usize;
        let mut scanned_spks = BTreeMap::new();

        loop {
            let spks = (0..batch_size)
                .map_while(|_| spks.next())
                .collect::<Vec<_>>();
            if spks.is_empty() {
                return Ok(scanned_spks);
            }

            let spk_histories = self
                .inner
                .batch_script_get_history(spks.iter().map(|(_, s)| s))?;

            for ((spk_index, spk), spk_history) in spks.into_iter().zip(spk_histories) {
                if spk_history.is_empty() {
                    scanned_spks.insert(spk_index, (spk, false));
                    unused_spk_count += 1;
                    if unused_spk_count > stop_gap {
                        return Ok(scanned_spks);
                    }
                    continue;
                } else {
                    scanned_spks.insert(spk_index, (spk, true));
                    unused_spk_count = 0;
                }

                for tx in spk_history {
                    let tx_pos = to_position(self.position_args(
                        tx.tx_hash,
                        match tx.height {
                            h if h <= 0 => TxHeight::Unconfirmed,
                            h => {
                                let h = h as u32;
                                if h > tip {
                                    TxHeight::Unconfirmed
                                } else {
                                    TxHeight::Confirmed(h)
                                }
                            }
                        },
                    ))?;
                    if let Err(failure) = update.insert_tx(tx.tx_hash, tx_pos) {
                        match failure {
                            sparse_chain::InsertTxError::TxTooHigh { .. } => {
                                unreachable!(
                                    "we should never encounter this as we ensured height <= tip"
                                );
                            }
                            sparse_chain::InsertTxError::TxMovedUnexpectedly { .. } => {
                                return Err(InternalError::Reorg);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Scan the blockchain (via electrum) for data specified by [`ScanParams`]. This returns a
    /// [`ElectrumUpdate`] which can be transformed into a [`KeychainScan`] after we find all the
    /// missing full transactions.
    ///
    /// Refer to [crate-level documentation] for more.
    ///
    /// [crate-level documentation]: crate
    pub fn scan<P, K, S>(
        &self,
        to_position: impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        local_chain: &BTreeMap<u32, BlockHash>,
        params: ScanParams<K, S>,
    ) -> Result<ElectrumUpdate<K, P>, Error>
    where
        P: ChainPosition,
        K: Ord + Clone,
        S: IntoIterator<Item = (u32, Script)>,
    {
        let mut request_spks = params
            .keychain_spks
            .into_iter()
            .map(|(k, s)| (k, s.into_iter()))
            .collect::<BTreeMap<K, _>>();
        let mut scanned_spks = BTreeMap::<(K, u32), (Script, bool)>::new();

        let update = loop {
            let mut update = self.prepare_update(&to_position, local_chain)?;

            if !request_spks.is_empty() {
                if !scanned_spks.is_empty() {
                    let mut scanned_spk_iter = scanned_spks
                        .iter()
                        .map(|(i, (spk, _))| (i.clone(), spk.clone()));
                    match self.populate_with_spks::<P, K, _, _>(
                        &to_position,
                        &mut update,
                        &mut scanned_spk_iter,
                        params.stop_gap,
                        params.batch_size,
                    ) {
                        Err(InternalError::Reorg) => continue,
                        Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                        Ok(mut spks) => scanned_spks.append(&mut spks),
                    };
                }
                for (keychain, keychain_spks) in &mut request_spks {
                    match self.populate_with_spks::<P, K, u32, _>(
                        &to_position,
                        &mut update,
                        keychain_spks,
                        params.stop_gap,
                        params.batch_size,
                    ) {
                        Err(InternalError::Reorg) => continue,
                        Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                        Ok(spks) => scanned_spks.extend(
                            spks.into_iter()
                                .map(|(spk_i, spk)| ((keychain.clone(), spk_i), spk)),
                        ),
                    };
                }
            }

            if !params.txids.is_empty() {
                match self.populate_with_txids(
                    &to_position,
                    &mut update,
                    params.txids.iter().cloned(),
                ) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_) => {}
                }
            }

            if !params.outpoints.is_empty() {
                match self.populate_with_outpoints(
                    &to_position,
                    &mut update,
                    params.outpoints.iter().cloned(),
                ) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_txs) => { /* [TODO] cache full txs to reduce bandwidth */ }
                }
            }

            // check for reorgs during scan process
            let our_tip = update
                .latest_checkpoint()
                .expect("update must have atleast one checkpoint");
            let server_blockhash = self.block_header(our_tip.height as usize)?.block_hash();
            if our_tip.hash != server_blockhash {
                continue; // reorg
            } else {
                break update;
            }
        };

        let last_active_index = request_spks
            .into_keys()
            .filter_map(|k| {
                scanned_spks
                    .range((k.clone(), u32::MIN)..=(k.clone(), u32::MAX))
                    .rev()
                    .find(|(_, (_, active))| *active)
                    .map(|((_, i), _)| (k, *i))
            })
            .collect::<BTreeMap<_, _>>();

        Ok(ElectrumUpdate {
            chain_update: update,
            last_active_indices: last_active_index,
        })
    }

    /// Convenience method to call [`scan`] without requiring a keychain.
    ///
    /// [`scan`]: ElectrumClient::scan
    pub fn scan_without_keychain<P, S>(
        &self,
        to_position: impl Fn(ChainPositionArgs<'_>) -> Result<P, Error>,
        local_chain: &BTreeMap<u32, BlockHash>,
        params: ScanParamsWithoutKeychain<S>,
    ) -> Result<ElectrumUpdate<(), P>, Error>
    where
        P: ChainPosition,
        S: IntoIterator<Item = Script>,
    {
        let spk_iter = params
            .spks
            .into_iter()
            .enumerate()
            .map(|(i, spk)| (i as u32, spk));
        let params = ScanParams {
            keychain_spks: [((), spk_iter)].into(),
            txids: params.txids,
            outpoints: params.outpoints,
            stop_gap: params.stop_gap,
            batch_size: params.batch_size,
        };
        self.scan(to_position, local_chain, params)
    }
}

/// Parameters for [`ElectrumClient::scan`].
pub struct ScanParams<K, S: IntoIterator<Item = (u32, Script)>> {
    /// Indexed script pubkeys of each keychain to scan transaction histories for.
    pub keychain_spks: BTreeMap<K, S>,
    /// Txids to scan for. The update will update the [`ChainPosition`]s for these.
    pub txids: Vec<Txid>,
    /// Outpoints to scan for. The update will try include the transaction that spends this outpoint
    /// alongside the transaction which contains this outpoint.
    pub outpoints: Vec<OutPoint>,
    /// The theshold number of [`ScanParams::keychain_spks`] that return empty histories before we
    /// stop scanning for `keychain_spks`.
    pub stop_gap: usize,
    /// The batch size to use for requests that can be batched.
    pub batch_size: usize,
}

impl<K, S: IntoIterator<Item = (u32, Script)>> Default for ScanParams<K, S> {
    fn default() -> Self {
        Self {
            keychain_spks: Default::default(),
            txids: Default::default(),
            outpoints: Default::default(),
            stop_gap: 10,
            batch_size: 10,
        }
    }
}

/// Parameters for [`ElectrumClient::scan_without_keychain`].
pub struct ScanParamsWithoutKeychain<S: IntoIterator<Item = Script>> {
    /// Miscellaneous script pubkeys to scan transaction histories for.
    pub spks: S,
    /// Txids to scan for. The update will update the [`ChainPosition`]s for these.
    pub txids: Vec<Txid>,
    /// Full transactions to scan for. The update will update the [`ChainPosition`]s for these.
    pub txs: Vec<Transaction>,
    /// Outpoints to scan for. The update will try include the transaction that spends this outpoint
    /// alongside the transaction which contains this outpoint.
    pub outpoints: Vec<OutPoint>,
    /// The theshold number of [`ScanParamsWithoutKeychain::spks`] that return empty histories
    /// before we stop scanning for `keychain_spks`.
    pub stop_gap: usize,
    /// The batch size to use for requests that can be batched.
    pub batch_size: usize,
}

impl<S: IntoIterator<Item = Script> + Default> Default for ScanParamsWithoutKeychain<S> {
    fn default() -> Self {
        Self {
            spks: Default::default(),
            txids: Default::default(),
            txs: Default::default(),
            outpoints: Default::default(),
            stop_gap: 10,
            batch_size: 10,
        }
    }
}

impl<S: IntoIterator<Item = Script>> ScanParamsWithoutKeychain<S> {
    /// Create [`ScanParamsWithoutKeychain`] with provided `spks` and default parameters.
    pub fn from_spks(spks: S) -> Self {
        Self {
            spks: spks,
            txids: Default::default(),
            txs: Default::default(),
            outpoints: Default::default(),
            stop_gap: 10,
            batch_size: 10,
        }
    }
}

/// The result of [`ElectrumClient::scan`].
pub struct ElectrumUpdate<K, P> {
    /// The internal [`SparseChain`] update.
    pub chain_update: SparseChain<P>,
    /// The last keychain script pubkey indices which had transaction histories.
    pub last_active_indices: BTreeMap<K, u32>,
}

impl<K, P> Default for ElectrumUpdate<K, P> {
    fn default() -> Self {
        Self {
            chain_update: Default::default(),
            last_active_indices: Default::default(),
        }
    }
}

impl<K, P> AsRef<SparseChain<P>> for ElectrumUpdate<K, P> {
    fn as_ref(&self) -> &SparseChain<P> {
        &self.chain_update
    }
}

impl<K: Ord + Clone + Debug, P: ChainPosition> ElectrumUpdate<K, P> {
    /// Return a list of missing full transactions that are required to [`inflate_update`].
    ///
    /// [`inflate_update`]: bdk_chain::chain_graph::ChainGraph::inflate_update
    pub fn find_missing_txids<T, G>(&self, graph: G) -> Vec<&Txid>
    where
        T: AsTransaction,
        G: AsRef<TxGraph<T>>,
    {
        self.chain_update
            .txids()
            .filter(|(_, txid)| graph.as_ref().get_tx(*txid).is_none())
            .map(|(_, txid)| txid)
            .collect()
    }

    /// Transform the [`ElectrumUpdate`] into a [`KeychainScan`] which can be applied to a
    /// `tracker`.
    ///
    /// This will fail if there are missing full transactions not provided via `new_txs`.
    pub fn into_keychain_scan<T, CG>(
        self,
        new_txs: Vec<T>,
        chain_graph: &CG,
    ) -> Result<KeychainScan<K, P, Cow<T>>, chain_graph::NewError<P>>
    where
        T: AsTransaction + Clone + Ord,
        CG: AsRef<ChainGraph<P, T>>,
    {
        Ok(KeychainScan {
            update: chain_graph
                .as_ref()
                .inflate_update(self.chain_update, new_txs)?,
            last_active_indices: self.last_active_indices,
        })
    }
}

#[derive(Debug)]
enum InternalError {
    ElectrumError(Error),
    Reorg,
}

impl From<electrum_client::Error> for InternalError {
    fn from(value: electrum_client::Error) -> Self {
        Self::ElectrumError(value.into())
    }
}
