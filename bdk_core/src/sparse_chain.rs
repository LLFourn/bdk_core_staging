use core::{
    fmt::{Debug, Display},
    ops::{Bound, RangeBounds},
};

use crate::{
    collections::*, tx_graph::TxGraph, BlockId, ConfirmationTime, FullTxOut, TxHeight, Vec,
};
use bitcoin::{hashes::Hash, BlockHash, OutPoint, Txid};

/// A [`SparseChain`] in which the [`ChainIndex`] is extended by a [`Timestamp`].
pub type TimestampedSparseChain = SparseChain<ConfirmationTime>;

/// This is a non-monotone structure that tracks relevant [`Txid`]s that are ordered by
/// [`ChainIndex`].
///
/// To "merge" two [`SparseChain`]s, one can calculate the [`ChangeSet`] by calling
/// [`Self::determine_changeset(update)`], and applying the [`ChangeSet`] via
/// [`Self::apply_changeset(changeset)`]. For convenience, one can do the above two steps as one via
/// [`Self::apply_update(update)`].
///
/// The generic `E` is used to extend the [`ChainIndex`], allowing for more definite ordering within
/// a given height.
#[derive(Clone, Debug)]
pub struct SparseChain<I: ChainIndex = TxHeight> {
    /// Block height to checkpoint data.
    checkpoints: BTreeMap<u32, BlockHash>,
    /// Ordered txids where order is enforced with the prepended height and index extension.
    ordered_txids: BTreeSet<(I, Txid)>,
    /// Confirmation heights of txids.
    txid_to_index: HashMap<Txid, I>,
    /// Limit number of checkpoints.
    checkpoint_limit: Option<usize>,
}

impl<I: ChainIndex> Default for SparseChain<I> {
    fn default() -> Self {
        Self {
            checkpoints: Default::default(),
            ordered_txids: Default::default(),
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
pub enum UpdateFailure<I: ChainIndex = TxHeight> {
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

impl<I: ChainIndex + core::fmt::Debug> core::fmt::Display for UpdateFailure<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotConnected(h) =>
                write!(f, "the checkpoints in the update could not be connected to the checkpoints in the chain, try include checkpoint of height {} to connect",
                    h),
            Self::InconsistentTx { inconsistent_txid, original_index, update_index } =>
                write!(f, "inconsistent update: first inconsistent tx is ({}) which had index ({:?}), but is ({:?}) in the update", 
                    inconsistent_txid, original_index, update_index),
        }
    }
}

#[cfg(feature = "std")]
impl<I: ChainIndex + core::fmt::Debug> std::error::Error for UpdateFailure<I> {}

impl<I: ChainIndex> SparseChain<I> {
    /// Creates a new chain from a list of block hashes and heights. The caller must guarantee they are in the same
    /// chain.
    pub fn from_checkpoints<C>(checkpoints: C) -> Self
    where
        C: IntoIterator<Item = BlockId>,
    {
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

    /// Return the associated index of a tx of txid (if any).
    pub fn tx_index(&self, txid: Txid) -> Option<I> {
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
                .ordered_txids
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

        for (&txid, &update_index) in &update.txid_to_index {
            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated, or originally unconfirmed)
            if let Some(&original_index) = self.txid_to_index.get(&txid) {
                if original_index.height() < TxHeight::Confirmed(invalid_from)
                    && original_index != update_index
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
                .ordered_txids
                // avoid invalidating mempool txids for initial change-set
                .range(
                    &(
                        I::min_ord_of_height(TxHeight::Confirmed(invalid_from)),
                        Txid::all_zeros(),
                    )
                        ..&(
                            I::min_ord_of_height(TxHeight::Unconfirmed),
                            Txid::all_zeros(),
                        ),
                )
                .map(|(key, txid)| (*txid, Change::new_removal(*key)))
                .collect(),
        };

        for (&height, &new_hash) in &update.checkpoints {
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

        for (txid, new_index) in &update.txid_to_index {
            let original_conf = self.txid_to_index.get(txid).cloned();

            let is_inaction = change_set
                .txids
                .entry(*txid)
                .and_modify(|change| change.to = Some(*new_index))
                .or_insert_with(|| Change::new(original_conf, Some(*new_index)))
                .is_inaction();

            if is_inaction {
                change_set.txids.remove(txid);
            }
        }

        Result::Ok(change_set)
    }

    /// Applies a new [`Update`] to the tracker.
    #[must_use]
    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok(changeset)
    }

    pub fn apply_changeset(&mut self, changeset: &ChangeSet<I>) {
        for (&height, change) in &changeset.checkpoints {
            let original_hash = match change.to {
                Some(to) => self.checkpoints.insert(height, to),
                None => self.checkpoints.remove(&height),
            };
            debug_assert_eq!(original_hash, change.from);
        }

        for (&txid, change) in &changeset.txids {
            let (changed, original_index) = match (change.from, change.to) {
                (None, None) => panic!("should not happen"),
                (None, Some(to)) => (
                    self.ordered_txids.insert((to, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
                (Some(from), None) => (
                    self.ordered_txids.remove(&(from, txid)),
                    self.txid_to_index.remove(&txid),
                ),
                (Some(from), Some(to)) => (
                    self.ordered_txids.insert((to, txid))
                        && self.ordered_txids.remove(&(from, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
            };
            debug_assert!(changed);
            debug_assert_eq!(original_index, change.from);
        }

        self.prune_checkpoints();
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) -> ChangeSet<I> {
        let txids = self
            .ordered_txids
            .range(
                &(
                    I::min_ord_of_height(TxHeight::Unconfirmed),
                    Txid::all_zeros(),
                )..,
            )
            .map(|(key, txid)| (*txid, Change::new_removal(*key)))
            .collect();

        let changeset = ChangeSet::<I> {
            txids,
            ..Default::default()
        };

        self.apply_changeset(&changeset);
        changeset
    }

    /// Insert an arbitary txid. This assumes that we have at least one checkpoint and the tx does
    /// not already exist in [`SparseChain`]. Returns a [`ChangeSet`] on success.
    pub fn insert_tx(&mut self, txid: Txid, index: I) -> Result<bool, InsertTxErr> {
        let new_height = index.height();

        let latest = self
            .checkpoints
            .keys()
            .last()
            .cloned()
            .map(TxHeight::Confirmed);

        if new_height.is_confirmed() && (latest.is_none() || new_height > latest.unwrap()) {
            return Err(InsertTxErr::TxTooHigh);
        }

        if let Some(&original_index) = self.txid_to_index.get(&txid) {
            if original_index.height().is_confirmed() && original_index != index {
                return Err(InsertTxErr::TxMoved);
            }

            return Ok(false);
        }

        self.txid_to_index.insert(txid, index);
        self.ordered_txids.insert((index, txid));

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
    ) -> impl DoubleEndedIterator<Item = &(I, Txid)> + ExactSizeIterator + '_ {
        self.ordered_txids.iter()
    }

    pub fn range_txids<R>(&self, range: R) -> impl DoubleEndedIterator<Item = &(I, Txid)> + '_
    where
        R: RangeBounds<(I, Txid)>,
    {
        let map_bound = |b: Bound<&(I, Txid)>| match b {
            Bound::Included(&(index, txid)) => Bound::Included((index, txid)),
            Bound::Excluded(&(index, txid)) => Bound::Excluded((index, txid)),
            Bound::Unbounded => Bound::Unbounded,
        };

        self.ordered_txids
            .range((map_bound(range.start_bound()), map_bound(range.end_bound())))
    }

    pub fn range_txids_by_index<R>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = &(I, Txid)> + '_
    where
        R: RangeBounds<I>,
    {
        let map_bound = |b: Bound<&I>, inc: Txid, exc: Txid| match b {
            Bound::Included(&index) => Bound::Included((index, inc)),
            Bound::Excluded(&index) => Bound::Excluded((index, exc)),
            Bound::Unbounded => Bound::Unbounded,
        };

        self.ordered_txids.range((
            map_bound(range.start_bound(), min_txid(), max_txid()),
            map_bound(range.end_bound(), max_txid(), min_txid()),
        ))
    }

    pub fn range_txids_by_height<R>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator<Item = &(I, Txid)> + '_
    where
        R: RangeBounds<TxHeight>,
    {
        let ord_it = |height, is_max| match is_max {
            true => I::max_ord_of_height(height),
            false => I::min_ord_of_height(height),
        };

        let map_bound = |b: Bound<&TxHeight>, inc: (bool, Txid), exc: (bool, Txid)| match b {
            Bound::Included(&h) => Bound::Included((ord_it(h, inc.0), inc.1)),
            Bound::Excluded(&h) => Bound::Excluded((ord_it(h, exc.0), exc.1)),
            Bound::Unbounded => Bound::Unbounded,
        };

        self.ordered_txids.range((
            map_bound(range.start_bound(), (false, min_txid()), (true, max_txid())),
            map_bound(range.end_bound(), (true, max_txid()), (false, min_txid())),
        ))
    }

    pub fn full_txout(&self, graph: &TxGraph, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        let chain_index = self.tx_index(outpoint.txid)?;

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
        Some(txids.iter().all(|&txid| self.tx_index(txid).is_none()))
    }
}

/// Represents the set of changes as result of a successful [`Update`].
#[derive(Debug, PartialEq)]
pub struct ChangeSet<I = TxHeight> {
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
                .and_modify(|change| change.to = new_change.clone().to)
                .or_insert_with(|| new_change)
                .is_inaction();

            if is_inaction {
                self.txids.remove_entry(&txid);
            }
        }

        Ok(self)
    }
}

impl<I> ChangeSet<I> {
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

impl<E: core::fmt::Debug> Display for MergeFailure<E> {
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
impl<D: core::fmt::Display + core::fmt::Debug> std::error::Error for MergeFailure<D> {}

/// Represents an index in which transactions are ordered by in [`SparseChain`].
///
/// [`ChainIndex`] implementations must be [`Ord`] by [`TxHeight`] first.
pub trait ChainIndex: Debug + Clone + Copy + Eq + PartialOrd + Ord + core::hash::Hash {
    /// Obtain the transaction height of the index.
    fn height(&self) -> TxHeight;

    /// Obtain the index's upper bound of a given height.
    fn max_ord_of_height(height: TxHeight) -> Self;

    /// Obtain the index's lower bound of a given height.
    fn min_ord_of_height(height: TxHeight) -> Self;
}

fn min_txid() -> Txid {
    Txid::from_inner([0x00; 32])
}

fn max_txid() -> Txid {
    Txid::from_inner([0xff; 32])
}

#[cfg(test)]
pub mod verify_chain_index {
    use alloc::vec::Vec;

    use super::ChainIndex;
    use crate::TxHeight;

    pub fn verify_chain_index<I: ChainIndex>(head_count: u32, tail_count: u32) {
        let values = (0..head_count)
            .chain(u32::MAX - tail_count..u32::MAX)
            .flat_map(|i| {
                [
                    I::min_ord_of_height(TxHeight::Confirmed(i)),
                    I::max_ord_of_height(TxHeight::Confirmed(i)),
                ]
            })
            .chain([
                I::min_ord_of_height(TxHeight::Unconfirmed),
                I::max_ord_of_height(TxHeight::Unconfirmed),
            ])
            .collect::<Vec<_>>();

        for i in 0..values.len() {
            for j in 0..values.len() {
                if i == j {
                    assert_eq!(values[i], values[j]);
                }
                if i < j {
                    assert!(values[i] <= values[j]);
                }
                if i > j {
                    assert!(values[i] >= values[j]);
                }
            }
        }
    }

    #[test]
    fn verify_tx_height() {
        verify_chain_index::<TxHeight>(1000, 1000);
    }
}
