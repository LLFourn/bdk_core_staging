use core::{
    fmt::Display,
    ops::{Bound, RangeBounds},
};

use crate::{collections::*, BlockId, TxGraph, Vec};
use bitcoin::{hashes::Hash, BlockHash, OutPoint, Transaction, TxOut, Txid};

#[derive(Clone, Debug)]
pub struct SparseChain<I = TxHeight> {
    /// Block height to checkpoint data.
    checkpoints: BTreeMap<u32, BlockHash>,
    /// Txids prepended by confirmation height.
    indexed_txids: BTreeSet<(I, Txid)>,
    /// Confirmation heights of txids.
    txid_to_index: HashMap<Txid, I>,
    /// Limit number of checkpoints.
    checkpoint_limit: Option<usize>,
}

/// The location of a transaction within the chain.
pub trait ChainIndex:
    Ord + Clone + Copy + core::fmt::Debug + PartialEq + Eq + core::hash::Hash
{
    fn height(&self) -> TxHeight;
    fn min_ord_for_height(height: TxHeight) -> Self;
    fn max_ord_for_height(height: TxHeight) -> Self;
}

impl ChainIndex for TxHeight {
    fn height(&self) -> TxHeight {
        *self
    }

    fn min_ord_for_height(height: TxHeight) -> Self {
        height
    }

    fn max_ord_for_height(height: TxHeight) -> Self {
        height
    }
}

impl<I> Default for SparseChain<I> {
    fn default() -> Self {
        Self {
            checkpoints: Default::default(),
            indexed_txids: Default::default(),
            txid_to_index: Default::default(),
            checkpoint_limit: Default::default(),
        }
    }
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
pub enum UpdateFailure<I> {
    /// The [`Update`] cannot be applied to this [`SparseChain`] because the chain suffix it
    /// represents did not connect to the existing chain. This error case contains the checkpoint
    /// height to include so that the chains can connect.
    NotConnected(u32),
    /// The [`Update`] canot be applied, because there are inconsistent tx states.
    /// This only reports the first inconsistency.
    InconsistentTx {
        inconsistent_txid: Txid,
        original_index: I,
        update_index: I,
    },
}

impl<I> core::fmt::Display for UpdateFailure<I>
where
    I: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotConnected(h) =>
                write!(f, "the checkpoints in the update could not be connected to the checkpoints in the chain, try include checkpoint of height {} to connect",
                    h),
            Self::InconsistentTx { inconsistent_txid, original_index: original_height, update_index: update_height } =>
                write!(f, "inconsistent update: first inconsistent tx is ({}) which had chain index {:?}, but is {:?} in the update",
                    inconsistent_txid, original_height, update_height),
        }
    }
}

#[cfg(feature = "std")]
impl<I: core::fmt::Debug> std::error::Error for UpdateFailure<I> {}

impl<I> SparseChain<I>
where
    I: ChainIndex,
{
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

    /// Returns the chain index of the tx and the data associated with it (if it exists).
    pub fn chain_index(&self, txid: Txid) -> Option<I> {
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
    pub fn initial_change_set(&self) -> ChangeSet<I> {
        ChangeSet {
            checkpoints: self
                .checkpoints
                .iter()
                .map(|(height, hash)| (*height, Change::new_insertion(*hash)))
                .collect(),
            txids: self
                .indexed_txids
                .iter()
                .map(|(index, txid)| (*txid, Change::new_insertion(*index)))
                .collect(),
        }
    }

    pub fn determine_changeset(&self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
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

        for &(update_index, txid) in &update.indexed_txids {
            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated, or originally unconfirmed)
            if let Some(&original_index) = self.txid_to_index.get(&txid) {
                if original_index.height() < TxHeight::Confirmed(invalid_from)
                    && update_index != original_index
                {
                    return Err(UpdateFailure::InconsistentTx {
                        inconsistent_txid: txid,
                        original_index,
                        update_index,
                    });
                }
            }
        }

        // create initial change-set, based on checkpoints and txids that are to be invalidated
        let mut change_set = ChangeSet::<I> {
            checkpoints: self
                .checkpoints
                .range(invalid_from..)
                .map(|(height, hash)| (*height, Change::new_removal(*hash)))
                .collect(),
            txids: self
                .indexed_txids
                // avoid invalidating mempool txids for initial change-set
                .range(
                    &(
                        I::min_ord_for_height(TxHeight::Confirmed(invalid_from)),
                        Txid::all_zeros(),
                    )
                        ..&(
                            I::min_ord_for_height(TxHeight::Unconfirmed),
                            Txid::all_zeros(),
                        ),
                )
                .map(|(chain_index, txid)| (*txid, Change::new_removal(*chain_index)))
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

        for &(new_index, txid) in &update.indexed_txids {
            let original_index = self.txid_to_index.get(&txid).cloned();

            let is_inaction = change_set
                .txids
                .entry(txid)
                .and_modify(|change| change.to = Some(new_index))
                .or_insert_with(|| Change::new(original_index, Some(new_index)))
                .is_inaction();

            if is_inaction {
                change_set.txids.remove(&txid);
            }
        }

        Ok(change_set)
    }

    /// Applies a new [`Update`] to the tracker.
    #[must_use]
    pub fn apply_update(
        &mut self,
        update: &SparseChain<I>,
    ) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(changeset.clone());
        Ok(changeset)
    }

    pub fn apply_changeset(&mut self, changeset: ChangeSet<I>) {
        for (&height, change) in &changeset.checkpoints {
            let original_hash = match change.to {
                Some(to) => self.checkpoints.insert(height, to),
                None => self.checkpoints.remove(&height),
            };
            debug_assert_eq!(original_hash, change.from);
        }

        for (txid, change) in changeset.txids {
            let (changed, original_height) = match (change.from, change.to) {
                (None, None) => panic!("should not happen"),
                (None, Some(to)) => (
                    self.indexed_txids.insert((to, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
                (Some(from), None) => (
                    self.indexed_txids.remove(&(from, txid)),
                    self.txid_to_index.remove(&txid),
                ),
                (Some(from), Some(to)) => (
                    self.indexed_txids.insert((to, txid))
                        && self.indexed_txids.remove(&(from, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
            };
            debug_assert!(changed);
            debug_assert_eq!(original_height, change.from);
        }

        self.prune_checkpoints();
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) -> ChangeSet<I> {
        let txids = self
            .indexed_txids
            .range(
                &(
                    I::min_ord_for_height(TxHeight::Unconfirmed),
                    Txid::all_zeros(),
                )..,
            )
            .map(|&(index, txid)| (txid, Change::new_removal(index)))
            .collect();

        let changeset = ChangeSet {
            txids,
            ..Default::default()
        };

        self.apply_changeset(changeset.clone());
        changeset
    }

    /// Insert an arbitary txid. This assumes that we have at least one checkpoint and the tx does
    /// not already exist in [`SparseChain`]. Returns a [`ChangeSet`] on success.
    pub fn insert_tx(&mut self, txid: Txid, chain_index: I) -> Result<bool, InsertTxErr> {
        let latest = self
            .checkpoints
            .keys()
            .last()
            .cloned()
            .map(TxHeight::Confirmed);

        let height = chain_index.height();

        if height.is_confirmed() && (latest.is_none() || height > latest.unwrap()) {
            return Err(InsertTxErr::TxTooHigh);
        }

        if let Some(old_chain_index) = self.txid_to_index.get(&txid) {
            if chain_index.height().is_confirmed() && &chain_index != old_chain_index {
                return Err(InsertTxErr::TxMoved);
            }

            return Ok(false);
        }

        self.txid_to_index.insert(txid, chain_index);
        self.indexed_txids.insert((chain_index, txid));

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
    ) -> impl DoubleEndedIterator<Item = (I, Txid)> + ExactSizeIterator + '_ {
        self.indexed_txids.iter().map(|(k, v)| (*k, *v))
    }

    pub fn txids_in_range<R: RangeBounds<I>>(&self, range: R) -> impl DoubleEndedIterator + '_ {
        fn map_bound<I: Clone + Copy>(bound: Bound<&I>) -> Bound<(I, Txid)> {
            match bound {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(x) => Bound::Included((*x, max_txid())),
                Bound::Excluded(x) => Bound::Excluded((*x, Txid::all_zeros())),
            }
        }

        self.indexed_txids
            .range((map_bound(range.start_bound()), map_bound(range.end_bound())))
            .map(|(k, v)| (*k, *v))
    }

    pub fn txids_in_height_range<R: RangeBounds<TxHeight>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator + '_ {
        fn map_bound<I: ChainIndex>(bound: Bound<&TxHeight>) -> Bound<(I, Txid)> {
            match bound {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(x) => Bound::Included((I::max_ord_for_height(*x), max_txid())),
                Bound::Excluded(x) => {
                    Bound::Excluded((I::min_ord_for_height(*x), Txid::all_zeros()))
                }
            }
        }

        self.indexed_txids
            .range((map_bound(range.start_bound()), map_bound(range.end_bound())))
            .map(|(k, v)| (*k, *v))
    }

    pub fn full_txout(&self, graph: &TxGraph, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        let chain_index = self.chain_index(outpoint.txid)?;

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
            chain_index,
            spent_by,
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
        Some(txids.iter().all(|&txid| self.chain_index(txid).is_none()))
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
#[derive(Debug, PartialEq, Clone)]
pub struct ChangeSet<I> {
    pub checkpoints: HashMap<u32, Change<BlockHash>>,
    pub txids: HashMap<Txid, Change<I>>,
}

impl<I> Default for ChangeSet<I> {
    fn default() -> Self {
        Self {
            checkpoints: Default::default(),
            txids: Default::default(),
        }
    }
}

impl<I: ChainIndex> ChangeSet<I> {
    pub fn merge(mut self, new_set: Self) -> Result<Self, MergeFailure<I>> {
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
                .and_modify(|change| change.to = new_change.to)
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
        fn fmt_opt<V: Display>(
            opt: &Option<V>,
            f: &mut core::fmt::Formatter<'_>,
        ) -> core::fmt::Result {
            match opt {
                Some(v) => v.fmt(f),
                None => "None".fmt(f),
            }
        }

        "(".fmt(f)?;
        fmt_opt(&self.from, f)?;
        " => ".fmt(f)?;
        fmt_opt(&self.to, f)?;
        ")".fmt(f)
    }
}

#[derive(Debug)]
pub enum MergeFailure<I> {
    Checkpoint(MergeConflict<u32, BlockHash>),
    Txid(MergeConflict<Txid, I>),
}

#[derive(Debug, Default)]
pub struct MergeConflict<K, V> {
    pub key: K,
    pub change: Change<V>,
    pub new_change: Change<V>,
}

impl<I: core::fmt::Debug> Display for MergeFailure<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            MergeFailure::Checkpoint(conflict) => write!(
                f,
                "merge conflict (checkpoint): height={}, original_change={}, merge_with={}",
                conflict.key, conflict.change, conflict.new_change
            ),
            MergeFailure::Txid(conflict) => write!(
                f,
                "merge conflict (tx): txid={}, original_change={:?}, merge_with={:?}",
                conflict.key, conflict.change, conflict.new_change
            ),
        }
    }
}

#[cfg(feature = "std")]
impl<I: core::fmt::Debug> std::error::Error for MergeFailure<I> {}

#[derive(Debug)]
pub enum SyncFailure {
    TxNotInGraph(Txid),
    TxNotInIndex(Txid),
    TxInconsistent {
        txid: Txid,
        original: Option<TxHeight>,
        change: Change<TxHeight>,
    },
}

impl Display for SyncFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // TODO: Proper error
        write!(f, "sync failure: {:?}", self)
    }
}

/// Represents the height in which a transaction is confirmed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TxHeight {
    Confirmed(u32),
    Unconfirmed,
}

impl AsRef<TxHeight> for TxHeight {
    fn as_ref(&self) -> &TxHeight {
        self
    }
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
pub struct FullTxOut<I = TxHeight> {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    /// The [`ChainIndex`] of the transaction with the output.
    pub chain_index: I,
    pub spent_by: Option<Txid>,
}

fn max_txid() -> Txid {
    Txid::from_inner([0xff; 32])
}
