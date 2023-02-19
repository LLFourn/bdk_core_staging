use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    marker::PhantomData,
    ops::Deref,
};

use bdk_chain::{
    bitcoin::{BlockHash, OutPoint, Script, Transaction, Txid},
    chain_graph::InflateAndUpdateError,
    keychain::{KeychainChangeSet, KeychainTracker},
    sparse_chain::{self, ChainPosition, SparseChain},
    BlockId, ConfirmationTime, TxHeight,
};
use electrum_client::{Client, ElectrumApi, GetHistoryRes};

pub trait ElectrumChainPosition: ChainPosition + Sized {
    fn get_chain_position(
        client: &ElectrumClient<Self>,
        txid: Txid,
        height: TxHeight,
    ) -> Result<Self, ElectrumError>;
}

impl ElectrumChainPosition for TxHeight {
    fn get_chain_position(
        _: &ElectrumClient<Self>,
        _: Txid,
        height: TxHeight,
    ) -> Result<TxHeight, ElectrumError> {
        Ok(height)
    }
}

impl ElectrumChainPosition for ConfirmationTime {
    fn get_chain_position(
        client: &ElectrumClient<Self>,
        _: Txid,
        height: TxHeight,
    ) -> Result<ConfirmationTime, ElectrumError> {
        Ok(match height {
            TxHeight::Confirmed(height) => {
                let time = client.block_header(height as _)?.time as u64;
                ConfirmationTime::Confirmed { height, time }
            }
            TxHeight::Unconfirmed => ConfirmationTime::Unconfirmed,
        })
    }
}

pub struct ElectrumClient<P> {
    inner: Client,
    pos_marker: PhantomData<P>,
}

impl<P> Deref for ElectrumClient<P> {
    type Target = Client;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<P> ElectrumClient<P>
where
    P: ChainPosition + ElectrumChainPosition,
{
    pub fn new(client: Client) -> Result<Self, ElectrumError> {
        Ok(Self {
            inner: client,
            pos_marker: Default::default(),
        })
    }

    pub fn get_chain_position(&self, txid: Txid, height: TxHeight) -> Result<P, ElectrumError> {
        P::get_chain_position(self, txid, height)
    }

    /// Fetch latest block height.
    pub fn get_tip(&self) -> Result<(u32, BlockHash), electrum_client::Error> {
        // TODO: unsubscribe when added to the client, or is there a better call to use here?
        Ok(self
            .inner
            .block_headers_subscribe()
            .map(|data| (data.height as u32, data.header.block_hash()))?)
    }

    /// Prepare an update sparsechain "template" based on the checkpoints of the `local_chain`.
    fn prepare_update(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
    ) -> Result<SparseChain<P>, ElectrumError> {
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
                    return self.prepare_update(local_chain);
                }
            }
        }

        Ok(update)
    }

    /// Check for reorgs during the sync process.
    fn reorg_check(&self, update: &mut SparseChain<P>) -> Result<(), InternalError> {
        let our_tip = update
            .latest_checkpoint()
            .expect("update must have atleast one checkpoint");
        let server_blockhash = self.block_header(our_tip.height as usize)?.block_hash();
        if our_tip.hash != server_blockhash {
            Err(InternalError::Reorg)
        } else {
            Ok(())
        }
    }

    /// Populate an update [`SparseChain`] with transactions (and associated block positions) that
    /// contain and spend the given `outpoints`.
    fn populate_with_outpoints(
        &self,
        update: &mut SparseChain<P>,
        outpoints: impl Iterator<Item = OutPoint>,
    ) -> Result<HashMap<Txid, Transaction>, InternalError> {
        let tip = update
            .latest_checkpoint()
            .expect("update must atleast have one checkpoint");
        let mut full_txs = HashMap::new();
        for outpoint in outpoints {
            for (tx_height, tx) in self.find_spending_tx(outpoint)? {
                let txid = tx.txid();
                full_txs.insert(txid, tx);
                if tx_height.is_confirmed() && tx_height > TxHeight::Confirmed(tip.height) {
                    continue;
                }
                let tx_pos = self.get_chain_position(txid, tx_height)?;
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
    fn populate_with_txids(
        &self,
        update: &mut SparseChain<P>,
        txids: impl Iterator<Item = Txid>,
    ) -> Result<(), InternalError> {
        let tip = update
            .latest_checkpoint()
            .expect("update must have atleast one checkpoint");
        for txid in txids {
            let tx_height = match self.get_txid_status(txid)? {
                Some(height) => height,
                None => continue,
            };
            if tx_height.is_confirmed() && tx_height > TxHeight::Confirmed(tip.height) {
                continue;
            }
            let tx_pos = self.get_chain_position(txid, tx_height)?;
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

    fn populate_with_txs<'a>(
        &self,
        update: &mut SparseChain<P>,
        txs: impl Iterator<Item = &'a Transaction>,
    ) -> Result<(), InternalError> {
        let tip = update
            .latest_checkpoint()
            .expect("update must have at least one checkpoint");
        for tx in txs {
            let tx_height = match self.get_tx_status(tx)? {
                Some(height) => height,
                None => continue,
            };
            if tx_height.is_confirmed() && tx_height > TxHeight::Confirmed(tip.height) {
                continue;
            }
            let txid = tx.txid();
            let tx_pos = self.get_chain_position(txid, tx_height)?;
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
    fn populate_with_spks<K, I, S>(
        &self,
        update: &mut SparseChain<P>,
        spks: &mut S,
        stop_gap: usize,
        batch_size: usize,
    ) -> Result<BTreeMap<I, (Script, bool)>, InternalError>
    where
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
                    let tx_pos = self.get_chain_position(
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
                    )?;
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

    fn find_spending_tx(
        &self,
        outpoint: OutPoint,
    ) -> Result<Vec<(TxHeight, Transaction)>, ElectrumError> {
        fn tx_height_from_result(res: &GetHistoryRes) -> TxHeight {
            match res.height {
                0 | -1 => TxHeight::Unconfirmed,
                h if h > 0 => TxHeight::Confirmed(h as _),
                invalid_h => panic!("unexpected height ({}) from electrum server", invalid_h),
            }
        }

        let tx = self.inner.transaction_get(&outpoint.txid)?;
        let txout = tx
            .output
            .get(outpoint.vout as usize)
            .ok_or(ElectrumError::InvalidOutPoint(outpoint))?;

        let mut spent_tx = None;
        let mut spending_tx = None;
        for item in self.inner.script_get_history(&txout.script_pubkey)? {
            let current_tx = self.inner.transaction_get(&item.tx_hash)?;
            if current_tx.txid() == tx.txid() {
                let height = tx_height_from_result(&item);
                spent_tx = Some((height, tx.clone()));
            }
            if current_tx
                .input
                .iter()
                .any(|txin| txin.previous_output == outpoint)
            {
                let height = tx_height_from_result(&item);
                spending_tx = Some((height, current_tx));
            }
            if spent_tx.is_none() && spending_tx.is_none() {
                break;
            }
        }

        Ok([spent_tx, spending_tx].into_iter().flatten().collect())
    }

    fn get_txid_status(&self, txid: Txid) -> Result<Option<TxHeight>, ElectrumError> {
        let tx = match self.inner.transaction_get(&txid) {
            Ok(tx) => tx,
            Err(electrum_client::Error::Protocol(_)) => return Ok(None),
            Err(other_err) => return Err(other_err.into()),
        };
        self.get_tx_status(&tx)
    }

    fn get_tx_status(&self, tx: &Transaction) -> Result<Option<TxHeight>, ElectrumError> {
        let txid = tx.txid();
        let spk = tx
            .output
            .get(0)
            .map(|txo| &txo.script_pubkey)
            .expect("tx must have an output");
        let tx_height = self
            .inner
            .script_get_history(spk)?
            .into_iter()
            .find(|r| r.tx_hash == txid)
            .map(|r| r.height);
        Ok(match tx_height {
            Some(h) if h > 0 => Some(TxHeight::Confirmed(h as _)),
            Some(_) => Some(TxHeight::Unconfirmed),
            None => None,
        })
    }

    /// Scan for a keychain tracker, and create an initial [`bdk_chain::sparse_chain::SparseChain`] update candidate.
    /// This will only contain [`Txid`]s in SparseChain, and no actual transaction data.
    ///
    /// User needs to fetch the required transaction data and create the final [`bdk_chain::keychain::KeychainChangeSet`] before applying it.
    pub fn scan<K, S>(
        &self,
        local_chain: &BTreeMap<u32, BlockHash>,
        params: ScanParams<K, S>,
    ) -> Result<ScanUpdate<K, P>, ElectrumError>
    where
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
            let mut update = self.prepare_update(local_chain)?;

            if !request_spks.is_empty() {
                if !scanned_spks.is_empty() {
                    let mut scanned_spk_iter = scanned_spks
                        .iter()
                        .map(|(i, (spk, _))| (i.clone(), spk.clone()));
                    match self.populate_with_spks::<K, _, _>(
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
                    match self.populate_with_spks::<K, u32, _>(
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

            if !params.arbitary_spks.is_empty() {
                let mut arbitary_spks = params
                    .arbitary_spks
                    .iter()
                    .enumerate()
                    .map(|(i, spk)| (i, spk.clone()));
                match self.populate_with_spks::<K, _, _>(
                    &mut update,
                    &mut arbitary_spks,
                    usize::MAX,
                    params.batch_size,
                ) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_) => {}
                }
            }

            if !params.txids.is_empty() {
                match self.populate_with_txids(&mut update, params.txids.iter().cloned()) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_) => {}
                }
            }

            if !params.full_txs.is_empty() {
                match self.populate_with_txs(&mut update, params.full_txs.iter()) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_) => {}
                }
            }

            if !params.outpoints.is_empty() {
                match self.populate_with_outpoints(&mut update, params.outpoints.iter().cloned()) {
                    Err(InternalError::Reorg) => continue,
                    Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                    Ok(_) => {}
                }
            }

            match self.reorg_check(&mut update) {
                Err(InternalError::Reorg) => continue,
                Err(InternalError::ElectrumError(e)) => return Err(e.into()),
                Ok(_) => break update,
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

        Ok(ScanUpdate {
            update,
            last_active_indices: last_active_index,
        })
    }
}

pub struct ScanParams<K, S: IntoIterator<Item = (u32, Script)>> {
    pub keychain_spks: BTreeMap<K, S>,
    pub arbitary_spks: Vec<Script>,
    pub txids: Vec<Txid>,
    pub full_txs: Vec<Transaction>,
    pub outpoints: Vec<OutPoint>,
    pub stop_gap: usize,
    pub batch_size: usize,
}

impl<K, S: IntoIterator<Item = (u32, Script)>> Default for ScanParams<K, S> {
    fn default() -> Self {
        Self {
            keychain_spks: Default::default(),
            arbitary_spks: Default::default(),
            txids: Default::default(),
            full_txs: Default::default(),
            outpoints: Default::default(),
            stop_gap: 0,
            batch_size: 1,
        }
    }
}

impl<K, S: IntoIterator<Item = (u32, Script)>> From<BTreeMap<K, S>> for ScanParams<K, S> {
    fn from(value: BTreeMap<K, S>) -> Self {
        Self {
            keychain_spks: value,
            ..Default::default()
        }
    }
}

pub struct ScanUpdate<K, P> {
    pub update: SparseChain<P>,
    pub last_active_indices: BTreeMap<K, u32>,
}

impl<K, P> Default for ScanUpdate<K, P> {
    fn default() -> Self {
        Self {
            update: Default::default(),
            last_active_indices: Default::default(),
        }
    }
}

impl<K, P> AsRef<SparseChain<P>> for ScanUpdate<K, P> {
    fn as_ref(&self) -> &SparseChain<P> {
        &self.update
    }
}

impl<K: Ord + Clone + Debug, P: ChainPosition> ScanUpdate<K, P> {
    pub fn apply(
        self,
        new_txs: Vec<Transaction>,
        tracker: &mut KeychainTracker<K, P>,
    ) -> Result<KeychainChangeSet<K, P, Transaction>, InflateAndUpdateError<P>> {
        tracker.set_lookahead_to_targets(self.last_active_indices);
        tracker.apply_sparsechain_update(self.update, new_txs)
    }
}

#[derive(Debug)]
pub enum ElectrumError {
    Client(electrum_client::Error),
    InvalidOutPoint(OutPoint),
}

impl core::fmt::Display for ElectrumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElectrumError::Client(e) => write!(f, "{}", e),
            ElectrumError::InvalidOutPoint(op) => {
                write!(f, "outpoint {}:{} does not exist", op.txid, op.vout)
            }
        }
    }
}

impl std::error::Error for ElectrumError {}

impl From<electrum_client::Error> for ElectrumError {
    fn from(e: electrum_client::Error) -> Self {
        Self::Client(e)
    }
}

impl From<serde_json::Error> for ElectrumError {
    fn from(value: serde_json::Error) -> Self {
        Self::Client(value.into())
    }
}

#[derive(Debug)]
enum InternalError {
    ElectrumError(ElectrumError),
    Reorg,
}

impl core::fmt::Display for InternalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalError::ElectrumError(e) => core::fmt::Display::fmt(e, f),
            InternalError::Reorg => write!(f, "reorg occured during update"),
        }
    }
}

impl std::error::Error for InternalError {}

impl From<ElectrumError> for InternalError {
    fn from(value: ElectrumError) -> Self {
        Self::ElectrumError(value)
    }
}

impl From<electrum_client::Error> for InternalError {
    fn from(value: electrum_client::Error) -> Self {
        Self::ElectrumError(value.into())
    }
}
