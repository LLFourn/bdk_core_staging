use core::{
    fmt::{Debug, Display},
    ops::{Bound, RangeBounds},
};

use crate::{collections::*, BlockId, TxGraph, Vec};
use bitcoin::{hashes::Hash, BlockHash, OutPoint, Transaction, TxOut, Txid};

#[derive(Clone, Debug, Default)]
pub struct SparseChain<D = ()> {
    /// Block height to checkpoint data.
    checkpoints: BTreeMap<u32, BlockHash>,
    /// Txids prepended by confirmation height.
    txid_by_height: BTreeSet<(TxData<D>, Txid)>,
    /// Confirmation heights of txids.
    txid_to_index: HashMap<Txid, TxData<D>>,
    /// Limit number of checkpoints.
    checkpoint_limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InsertTxErr {
    TxTooHigh,
    TxMoved,
}

impl Display for InsertTxErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InsertTxErr {}

#[derive(Clone, Debug, PartialEq)]
pub enum InsertCheckpointErr {
    HashNotMatching,
}

impl Display for InsertCheckpointErr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InsertCheckpointErr {}

/// Represents an update failure of [`SparseChain`].
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure {
    /// The [`Update`] cannot be applied to this [`SparseChain`] because the chain suffix it
    /// represents did not connect to the existing chain. This error case contains the checkpoint
    /// height to include so that the chains can connect.
    NotConnected(u32),
    /// The [`Update`] canot be applied, because there are inconsistent tx states.
    /// This only reports the first inconsistency.
    InconsistentTx {
        inconsistent_txid: Txid,
        original_height: TxHeight,
        update_height: TxHeight,
    },
}

impl core::fmt::Display for UpdateFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotConnected(h) =>
                write!(f, "the checkpoints in the update could not be connected to the checkpoints in the chain, try include checkpoint of height {} to connect",
                    h),
            Self::InconsistentTx { inconsistent_txid, original_height, update_height } =>
                write!(f, "inconsistent update: first inconsistent tx is ({}) which had confirmation height ({}), but is ({}) in the update", 
                    inconsistent_txid, original_height, update_height),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UpdateFailure {}

impl<D: Clone + Debug + Default + Ord> SparseChain<D> {
    /// Creates a new chain from a list of blocks. The caller must guarantee they are in the same
    /// chain.
    pub fn from_checkpoints(checkpoints: impl IntoIterator<Item = BlockId>) -> Self {
        let mut chain = Self::default();
        chain.checkpoints = checkpoints
            .into_iter()
            .map(|block_id| block_id.into())
            .collect();
        chain
    }
    /// Get the BlockId for the last known tip.
    pub fn latest_checkpoint(&self) -> Option<BlockId> {
        self.checkpoints
            .iter()
            .last()
            .map(|(&height, &hash)| BlockId { height, hash })
    }

    /// Get the checkpoint id at the given height if it exists
    pub fn checkpoint_at(&self, height: u32) -> Option<BlockId> {
        self.checkpoints
            .get(&height)
            .map(|&hash| BlockId { height, hash })
    }

    /// Return the associated data of a tx of txid (if any).
    pub fn tx_data(&self, txid: Txid) -> Option<TxData<D>> {
        self.txid_to_index.get(&txid).cloned()
    }

    /// Return an iterator over all checkpoints, in descensing order.
    pub fn checkpoints(&self) -> &BTreeMap<u32, BlockHash> {
        &self.checkpoints
    }

    /// Return an iterator over the checkpoint locations in a height range, in ascending height order.
    pub fn range_checkpoints(
        &self,
        range: impl RangeBounds<u32>,
    ) -> impl DoubleEndedIterator<Item = BlockId> + '_ {
        self.checkpoints
            .range(range)
            .map(|(&height, &hash)| BlockId { height, hash })
    }

    /// Derives a [`ChangeSet`] that could be applied to an empty index.
    pub fn initial_change_set(&self) -> ChangeSet<D> {
        ChangeSet {
            checkpoints: self
                .checkpoints
                .iter()
                .map(|(height, hash)| (*height, Change::new_insertion(*hash)))
                .collect(),
            txids: self
                .txid_by_height
                .iter()
                .map(|(data, txid)| (*txid, Change::new_insertion(data.clone())))
                .collect(),
        }
    }

    pub fn determine_changeset(&self, update: &Self) -> Result<ChangeSet<D>, UpdateFailure> {
        let agreement_point = update
            .checkpoints
            .iter()
            .rev()
            .find(|&(height, hash)| self.checkpoints.get(height) == Some(hash))
            .map(|(&h, _)| h);

        let last_update_cp = update.checkpoints.iter().last().map(|(&h, _)| h);

        // checkpoints of this height and after are to be invalidated
        let invalid_from = if last_update_cp.is_none() || last_update_cp == agreement_point {
            // if agreement point is the last update checkpoint, or there is no update checkpoints,
            // no invalidation is required
            u32::MAX
        } else {
            agreement_point.map(|h| h + 1).unwrap_or(0)
        };

        // the first checkpoint of the sparsechain to invalidate (if any)
        let first_invalid = self
            .checkpoints
            .range(invalid_from..)
            .next()
            .map(|(&h, _)| h);

        // the first checkpoint to invalidate (if any) should be represented in the update
        if let Some(first_invalid) = first_invalid {
            if !update.checkpoints.contains_key(&first_invalid) {
                return Err(UpdateFailure::NotConnected(first_invalid));
            }
        }

        for (tx_data, txid) in &update.txid_by_height {
            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated, or originally unconfirmed)
            if let Some(old_data) = self.txid_to_index.get(txid) {
                if old_data.height < TxHeight::Confirmed(invalid_from)
                    && tx_data.height != old_data.height
                {
                    return Err(UpdateFailure::InconsistentTx {
                        inconsistent_txid: *txid,
                        original_height: old_data.height,
                        update_height: tx_data.height,
                    });
                }
            }
        }

        // create initial change-set, based on checkpoints and txids that are to be invalidated
        let mut change_set = ChangeSet {
            checkpoints: self
                .checkpoints
                .range(invalid_from..)
                .map(|(height, hash)| (*height, Change::new_removal(*hash)))
                .collect(),
            txids: self
                .txid_by_height
                // avoid invalidating mempool txids for initial change-set
                .range(
                    &(TxHeight::Confirmed(invalid_from).into(), Txid::all_zeros())
                        ..&(TxHeight::Unconfirmed.into(), Txid::all_zeros()),
                )
                .map(|(height, txid)| (*txid, Change::new_removal(height.clone())))
                .collect(),
        };

        for (&height, &new_hash) in update.checkpoints.iter() {
            let original_hash = self.checkpoints.get(&height).cloned();

            let is_inaction = change_set
                .checkpoints
                .entry(height)
                .and_modify(|change| change.to = Some(new_hash))
                .or_insert_with(|| Change::new(original_hash, Some(new_hash)))
                .is_inaction();

            if is_inaction {
                change_set.checkpoints.remove(&height);
            }
        }

        for (new_conf, txid) in &update.txid_by_height {
            let original_conf = self.txid_to_index.get(txid).cloned();

            let is_inaction = change_set
                .txids
                .entry(*txid)
                .and_modify(|change| change.to = Some(new_conf.clone()))
                .or_insert_with(|| Change::new(original_conf, Some(new_conf.clone())))
                .is_inaction();

            if is_inaction {
                change_set.txids.remove(txid);
            }
        }

        Result::Ok(change_set)
    }

    /// Applies a new [`Update`] to the tracker.
    #[must_use]
    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<D>, UpdateFailure> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok(changeset)
    }

    pub fn apply_changeset(&mut self, changeset: &ChangeSet<D>) {
        for (&height, change) in &changeset.checkpoints {
            let original_hash = match change.to {
                Some(to) => self.checkpoints.insert(height, to),
                None => self.checkpoints.remove(&height),
            };
            debug_assert_eq!(original_hash, change.from);
        }

        for (&txid, change) in &changeset.txids {
            let (changed, original_height) = match (&change.from, &change.to) {
                (None, None) => panic!("should not happen"),
                (None, Some(to)) => (
                    self.txid_by_height.insert((to.clone(), txid)),
                    self.txid_to_index.insert(txid, to.clone()),
                ),
                (Some(from), None) => (
                    self.txid_by_height.remove(&(from.clone(), txid)),
                    self.txid_to_index.remove(&txid),
                ),
                (Some(from), Some(to)) => (
                    self.txid_by_height.insert((to.clone(), txid))
                        && self.txid_by_height.remove(&(from.clone(), txid)),
                    self.txid_to_index.insert(txid, to.clone()),
                ),
            };
            debug_assert!(changed);
            debug_assert_eq!(original_height, change.from);
        }

        self.prune_checkpoints();
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) -> ChangeSet<D> {
        let txids = self
            .txid_by_height
            .range(&(TxHeight::Unconfirmed.into(), Txid::all_zeros())..)
            .map(|(data, txid)| (*txid, Change::new_removal(data.clone())))
            .collect();

        let changeset = ChangeSet {
            txids,
            ..Default::default()
        };

        self.apply_changeset(&changeset);
        changeset
    }

    /// Insert an arbitary txid. This assumes that we have at least one checkpoint and the tx does
    /// not already exist in [`SparseChain`]. Returns a [`ChangeSet`] on success.
    pub fn insert_tx(&mut self, txid: Txid, height: TxHeight) -> Result<bool, InsertTxErr> {
        self.insert_tx_with_additional_data(txid, height.into())
    }

    pub fn insert_tx_with_additional_data(
        &mut self,
        txid: Txid,
        additional_data: TxData<D>,
    ) -> Result<bool, InsertTxErr> {
        let latest = self
            .checkpoints
            .keys()
            .last()
            .cloned()
            .map(TxHeight::Confirmed);

        if additional_data.height.is_confirmed()
            && (latest.is_none() || additional_data.height > latest.unwrap())
        {
            return Err(InsertTxErr::TxTooHigh);
        }

        if let Some(old_data) = self.txid_to_index.get(&txid) {
            if old_data.height.is_confirmed() && old_data.height != additional_data.height {
                return Err(InsertTxErr::TxMoved);
            }

            return Ok(false);
        }

        self.txid_to_index.insert(txid, additional_data.clone());
        self.txid_by_height.insert((additional_data, txid));

        Ok(true)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        if let Some(&old_hash) = self.checkpoints.get(&block_id.height) {
            if old_hash != block_id.hash {
                return Err(InsertCheckpointErr::HashNotMatching);
            }

            return Ok(false);
        }

        self.checkpoints.insert(block_id.height, block_id.hash);
        self.prune_checkpoints();
        Ok(true)
    }

    pub fn iter_txids(
        &self,
    ) -> impl DoubleEndedIterator<Item = (TxData<D>, Txid)> + ExactSizeIterator + '_ {
        self.txid_by_height.iter().map(|(k, v)| (k.clone(), *v))
    }

    pub fn iter_txids_in_height_range<R: RangeBounds<TxHeight>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator + '_ {
        fn map_bound<D: Clone + Default>(bound: Bound<&TxHeight>) -> Bound<(TxData<D>, Txid)> {
            match bound {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(x) => Bound::Included((x.clone().into(), Txid::all_zeros())),
                Bound::Excluded(x) => Bound::Excluded((x.clone().into(), Txid::all_zeros())),
            }
        }

        self.txid_by_height
            .range((map_bound(range.start_bound()), map_bound(range.end_bound())))
    }
    pub fn full_txout(&self, graph: &TxGraph, outpoint: OutPoint) -> Option<FullTxOut<D>> {
        let data = self.tx_data(outpoint.txid)?;

        let txout = graph.txout(outpoint).cloned()?;

        let spent_by = graph
            .outspend(outpoint)
            .map(|txid_map| {
                // find txids
                let txids = txid_map
                    .iter()
                    .filter(|&txid| self.txid_to_index.contains_key(txid))
                    .collect::<Vec<_>>();
                debug_assert!(txids.len() <= 1, "conflicting txs in sparse chain");
                txids.get(0).cloned()
            })
            .flatten()
            .cloned();

        Some(FullTxOut {
            outpoint,
            txout,
            height: data.height,
            spent_by,
            additional_data: data.additional,
        })
    }

    pub fn set_checkpoint_limit(&mut self, limit: Option<usize>) {
        self.checkpoint_limit = limit;
        self.prune_checkpoints();
    }

    fn prune_checkpoints(&mut self) -> Option<BTreeMap<u32, BlockHash>> {
        let limit = self.checkpoint_limit?;

        // find last height to be pruned
        let last_height = *self.checkpoints.keys().rev().nth(limit)?;
        // first height to be kept
        let keep_height = last_height + 1;

        let mut split = self.checkpoints.split_off(&keep_height);
        core::mem::swap(&mut self.checkpoints, &mut split);

        Some(split)
    }

    /// Determines whether outpoint is spent or not. Returns `None` when outpoint does not exist in
    /// graph.
    pub fn is_unspent(&self, graph: &TxGraph, outpoint: OutPoint) -> Option<bool> {
        let txids = graph.outspend(outpoint)?;
        Some(txids.iter().all(|&txid| self.tx_data(txid).is_none()))
    }
}

#[derive(Debug, Clone)]
pub enum TxKey {
    Txid(Txid),
    Tx(Transaction),
}

impl TxKey {
    pub fn txid(&self) -> Txid {
        match self {
            TxKey::Txid(txid) => *txid,
            TxKey::Tx(tx) => tx.txid(),
        }
    }
}

impl From<Txid> for TxKey {
    fn from(txid: Txid) -> Self {
        Self::Txid(txid)
    }
}

impl From<Transaction> for TxKey {
    fn from(tx: Transaction) -> Self {
        Self::Tx(tx)
    }
}

impl PartialEq for TxKey {
    fn eq(&self, other: &Self) -> bool {
        self.txid() == other.txid()
    }
}

impl Eq for TxKey {}

impl PartialOrd for TxKey {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TxKey {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.txid().cmp(&other.txid())
    }
}

impl core::hash::Hash for TxKey {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.txid().hash(state)
    }
}

/// Represents the set of changes as result of a successful [`Update`].
#[derive(Debug, Default, PartialEq)]
pub struct ChangeSet<D = ()> {
    pub checkpoints: HashMap<u32, Change<BlockHash>>,
    pub txids: HashMap<Txid, Change<TxData<D>>>,
}

impl<D: Clone + PartialEq> ChangeSet<D> {
    pub fn merge(mut self, new_set: Self) -> Result<Self, MergeFailure<D>> {
        for (height, new_change) in new_set.checkpoints {
            if let Some(change) = self.checkpoints.get(&height) {
                if change.to != new_change.from {
                    return Err(MergeFailure::Checkpoint(MergeConflict {
                        key: height,
                        change: change.clone(),
                        new_change,
                    }));
                }
            }

            let is_inaction = self
                .checkpoints
                .entry(height)
                .and_modify(|change| change.to = new_change.to)
                .or_insert_with(|| new_change.clone())
                .is_inaction();

            if is_inaction {
                self.checkpoints.remove_entry(&height);
            }
        }

        for (txid, new_change) in new_set.txids {
            if let Some(change) = self.txids.get(&txid) {
                if change.to != new_change.from {
                    return Err(MergeFailure::Txid(MergeConflict {
                        key: txid,
                        change: change.clone(),
                        new_change,
                    }));
                }
            }

            let is_inaction = self
                .txids
                .entry(txid)
                .and_modify(|change| change.to = new_change.clone().to)
                .or_insert_with(|| new_change.clone())
                .is_inaction();

            if is_inaction {
                self.txids.remove_entry(&txid);
            }
        }

        Ok(self)
    }

    pub fn tx_additions(&self) -> impl Iterator<Item = Txid> + '_ {
        self.txids
            .iter()
            .filter_map(|(txid, change)| match (&change.from, &change.to) {
                (None, Some(_)) => Some(*txid),
                _ => None,
            })
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Change<V> {
    pub from: Option<V>,
    pub to: Option<V>,
}

impl<V> Change<V> {
    pub fn new(from: Option<V>, to: Option<V>) -> Self {
        Self { from, to }
    }

    pub fn new_removal(v: V) -> Self {
        Self {
            from: Some(v),
            to: None,
        }
    }

    pub fn new_insertion(v: V) -> Self {
        Self {
            from: None,
            to: Some(v),
        }
    }

    pub fn new_alteration(from: V, to: V) -> Self {
        Self {
            from: Some(from),
            to: Some(to),
        }
    }
}

impl<V: PartialEq> Change<V> {
    pub fn is_inaction(&self) -> bool {
        self.from == self.to
    }
}

impl<V: Display> Display for Change<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // use core::fmt::Display as display_fmt;

        fn fmt_opt<V: Display>(
            opt: &Option<V>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            match opt {
                Some(v) => v.fmt(f),
                None => Display::fmt("None", f),
            }
        }

        Display::fmt("(", f)?;
        fmt_opt(&self.from, f)?;
        Display::fmt(" => ", f)?;
        fmt_opt(&self.to, f)?;
        Display::fmt(")", f)
    }
}

#[derive(Debug)]
pub enum MergeFailure<D> {
    Checkpoint(MergeConflict<u32, BlockHash>),
    Txid(MergeConflict<Txid, TxData<D>>),
}

#[derive(Debug, Default)]
pub struct MergeConflict<K, V> {
    pub key: K,
    pub change: Change<V>,
    pub new_change: Change<V>,
}

impl<D: core::fmt::Display> Display for MergeFailure<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MergeFailure::Checkpoint(conflict) => write!(
                f,
                "merge conflict (checkpoint): height={}, original_change={}, merge_with={}",
                conflict.key, conflict.change, conflict.new_change
            ),
            MergeFailure::Txid(conflict) => write!(
                f,
                "merge conflict (tx): txid={}, original_change={}, merge_with={}",
                conflict.key, conflict.change, conflict.new_change
            ),
        }
    }
}

#[cfg(feature = "std")]
impl<D: core::fmt::Display + core::fmt::Debug> std::error::Error for MergeFailure<D> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TxData<D = ()> {
    pub height: TxHeight,
    pub additional: D,
}

impl<D: Display> Display for TxData<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::write!(f, "data( height={}, {} )", self.height, self.additional)
    }
}

impl<D: Default> From<TxHeight> for TxData<D> {
    fn from(height: TxHeight) -> Self {
        Self {
            height,
            additional: D::default(),
        }
    }
}

impl<D: Default> From<Option<u32>> for TxData<D> {
    fn from(height: Option<u32>) -> Self {
        Self {
            height: height.into(),
            additional: D::default(),
        }
    }
}

/// Represents the height in which a transaction is confirmed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TxHeight {
    Confirmed(u32),
    Unconfirmed,
}

impl Display for TxHeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Confirmed(h) => core::write!(f, "confirmed_at({})", h),
            Self::Unconfirmed => core::write!(f, "unconfirmed"),
        }
    }
}

impl From<Option<u32>> for TxHeight {
    fn from(opt: Option<u32>) -> Self {
        match opt {
            Some(h) => Self::Confirmed(h),
            None => Self::Unconfirmed,
        }
    }
}

impl From<TxHeight> for Option<u32> {
    fn from(height: TxHeight) -> Self {
        match height {
            TxHeight::Confirmed(h) => Some(h),
            TxHeight::Unconfirmed => None,
        }
    }
}

impl TxHeight {
    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }
}

/// A `TxOut` with as much data as we can retreive about it
#[derive(Debug, Clone, PartialEq)]
pub struct FullTxOut<D> {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub height: TxHeight,
    pub spent_by: Option<Txid>,
    pub additional_data: D,
}
