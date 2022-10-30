use core::{
    fmt::Display,
    ops::{Bound, RangeBounds},
};

use crate::{collections::*, BlockId, TxGraph, Vec};
use bitcoin::{hashes::Hash, Block, BlockHash, OutPoint, Transaction, TxOut, Txid};

#[derive(Clone, Debug, Default)]
pub struct SparseChain {
    /// Block height to checkpoint data.
    checkpoints: BTreeMap<u32, BlockHash>,
    /// Txids prepended by confirmation height.
    txid_by_height: BTreeSet<(TxHeight, Txid)>,
    /// Confirmation heights of txids.
    txid_to_index: HashMap<Txid, TxHeight>,
    /// Limit number of checkpoints.
    checkpoint_limit: Option<usize>,
}

/// Represents an update failure of [`SparseChain`].
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure {
    /// The [`Update`] is total bogus. Cannot be applied to any [`SparseChain`].
    Bogus(BogusReason),

    /// The [`Update`] cannot be applied to this [`SparseChain`] because the chain suffix it
    /// represents did not connect to the existing chain.
    /// TODO: Add last_valid and invalid_from?
    NotConnected,
    /// The [`Update`] canot be applied, because there are inconsistent tx states.
    /// This only reports the first inconsistency.
    Inconsistent {
        inconsistent_txid: Txid,
        original_height: TxHeight,
        update_height: TxHeight,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum BogusReason {
    /// At least one `txid` has a confirmation height greater than `new_tip`.
    TxHeightGreaterThanTip {
        txid: Txid,
        tx_height: u32,
        tip_height: u32,
    },
    /// There were no checkpoints in the update
    EmptyCheckpoints,
}

impl core::fmt::Display for UpdateFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bogus(reason) => {
                write!(f, "bogus update: ")?;
                match reason {
                    BogusReason::EmptyCheckpoints => write!(f, "the checkpoints in the update were empty"),
                    BogusReason::TxHeightGreaterThanTip { txid, tx_height, tip_height, } =>
                        write!(f, "tx ({}) confirmation height ({}) is greater than new_tip height ({})",
                            txid, tx_height, tip_height),
                }
            },
            Self::NotConnected  => write!(f, "the checkpoints in the update could not be connected to the checkpoints in the chain"),
            Self::Inconsistent { inconsistent_txid, original_height, update_height } =>
                write!(f, "inconsistent update: first inconsistent tx is ({}) which had confirmation height ({}), but is ({}) in the update", 
                    inconsistent_txid, original_height, update_height),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UpdateFailure {}

impl SparseChain {
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

    /// Return height of tx (if any).
    pub fn transaction_height(&self, txid: Txid) -> Option<TxHeight> {
        self.txid_to_index.get(&txid).cloned()
    }

    /// Return an iterator over all checkpoints, in descensing order.
    pub fn checkpoints(&self) -> impl DoubleEndedIterator<Item = BlockId> + ExactSizeIterator + '_ {
        self.checkpoints
            .iter()
            .rev()
            .map(|(&height, &hash)| BlockId { height, hash })
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
    pub fn initial_change_set(&self) -> ChangeSet {
        ChangeSet {
            checkpoints: self
                .checkpoints
                .iter()
                .map(|(height, hash)| (*height, Change::new_insertion(*hash)))
                .collect(),
            txids: self
                .txid_by_height
                .iter()
                .map(|(height, txid)| (*txid, Change::new_insertion(*height)))
                .collect(),
        }
    }

    pub fn determine_changeset(&self, update: Update) -> Result<ChangeSet, UpdateFailure> {
        let last_valid = update
            .checkpoints
            .iter()
            .rev()
            .find(|&(height, hash)| self.checkpoints.get(height) == Some(hash))
            .map(|(&h, _)| h);

        // checkpoints of this height and after are to be invalidated
        let invalid_from = match update.checkpoints.is_empty() {
            true => u32::MAX,
            false => last_valid.map(|h| h + 1).unwrap_or(0),
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
                return Err(UpdateFailure::NotConnected);
            }
        }

        // the new checkpoint tip introduced by the update (if any)
        let new_tip = update.checkpoints.iter().last().map(|(&h, _)| h);

        for (&txid, &tx_height) in &update.txids {
            // ensure tx height does not surpass update's tip
            // the exception is that unconfirmed transactions are always allowed
            if let (Some(tip_height), TxHeight::Confirmed(tx_height)) = (new_tip, tx_height) {
                if tip_height < tx_height {
                    return Err(UpdateFailure::Bogus(BogusReason::TxHeightGreaterThanTip {
                        txid,
                        tx_height,
                        tip_height,
                    }));
                }
            }

            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated, or originally unconfirmed)
            if let Some(&old_height) = self.txid_to_index.get(&txid) {
                if old_height < TxHeight::Confirmed(invalid_from) && tx_height != old_height {
                    return Err(UpdateFailure::Inconsistent {
                        inconsistent_txid: txid,
                        original_height: old_height,
                        update_height: tx_height,
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
                    &(TxHeight::Confirmed(invalid_from), Txid::all_zeros())
                        ..&(TxHeight::Unconfirmed, Txid::all_zeros()),
                )
                .map(|(height, txid)| (*txid, Change::new_removal(*height)))
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

        for (txid, new_conf) in update.txids {
            let original_conf = self.txid_to_index.get(&txid).cloned();

            let is_inaction = change_set
                .txids
                .entry(txid)
                .and_modify(|change| change.to = Some(new_conf))
                .or_insert_with(|| Change::new(original_conf, Some(new_conf)))
                .is_inaction();

            if is_inaction {
                change_set.txids.remove(&txid);
            }
        }

        Result::Ok(change_set)
    }

    /// Applies a new [`Update`] to the tracker.
    #[must_use]
    pub fn apply_update(&mut self, update: Update) -> Result<ChangeSet, UpdateFailure> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok(changeset)
    }

    pub fn apply_changeset(&mut self, changeset: &ChangeSet) {
        for (&height, change) in &changeset.checkpoints {
            let original_hash = match change.to {
                Some(to) => self.checkpoints.insert(height, to),
                None => self.checkpoints.remove(&height),
            };
            debug_assert_eq!(original_hash, change.from);
        }

        for (&txid, change) in &changeset.txids {
            let (changed, original_height) = match (change.from, change.to) {
                (None, None) => panic!("should not happen"),
                (None, Some(to)) => (
                    self.txid_by_height.insert((to, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
                (Some(from), None) => (
                    self.txid_by_height.remove(&(from, txid)),
                    self.txid_to_index.remove(&txid),
                ),
                (Some(from), Some(to)) => (
                    self.txid_by_height.insert((to, txid))
                        && self.txid_by_height.remove(&(from, txid)),
                    self.txid_to_index.insert(txid, to),
                ),
            };
            debug_assert!(changed);
            debug_assert_eq!(original_height, change.from);
        }

        self.prune_checkpoints();
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) -> ChangeSet {
        let txids = self
            .txid_by_height
            .range(&(TxHeight::Unconfirmed, Txid::all_zeros())..)
            .map(|&(_, txid)| (txid, Change::new_removal(TxHeight::Unconfirmed)))
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
    /// TODO: Fix the error case!!
    pub fn insert_tx(&mut self, txid: Txid, height: TxHeight) -> Result<ChangeSet, UpdateFailure> {
        let update = Update {
            txids: [(txid, height)].into(),
            checkpoints: self
                .latest_checkpoint()
                .iter()
                .map(|cp| (cp.height, cp.hash))
                .collect(),
        };

        self.apply_update(update)
    }

    /// Inserts a checkpoint at any height in the chain. If it conflicts with an exisitng checkpoint
    /// the existing one will be invalidated and removed.
    pub fn apply_checkpoint(&mut self, checkpoint: BlockId) -> ChangeSet {
        self.apply_block_txs(checkpoint, core::iter::empty())
            .expect("cannot fail")
    }

    /// Apply a block with a caller provided height.
    pub fn apply_block_with_height(
        &mut self,
        block: &Block,
        height: u32,
        mut filter: impl FnMut(&Transaction) -> bool,
    ) -> Result<ChangeSet, UpdateFailure> {
        self.apply_block_txs(
            BlockId {
                height,
                hash: block.block_hash(),
            },
            block
                .txdata
                .iter()
                .filter(|tx| filter(tx))
                .map(|tx| tx.txid()),
        )
    }

    /// Apply a bitcoin block bip34 compliant block
    ///
    /// ## Panics
    ///
    /// Panics if the block is not a bip34 compliant block
    pub fn apply_block(
        &mut self,
        block: &Block,
        filter: impl FnMut(&Transaction) -> bool,
    ) -> Result<ChangeSet, UpdateFailure> {
        self.apply_block_with_height(
            block,
            block.bip34_block_height().expect("valid bip34 block") as u32,
            filter,
        )
    }

    /// Apply transactions that are all confirmed in a given block
    pub fn apply_block_txs(
        &mut self,
        checkpoint: BlockId,
        txs: impl IntoIterator<Item = Txid>,
    ) -> Result<ChangeSet, UpdateFailure> {
        let mut checkpoints = self
            .checkpoints
            .iter()
            .rev()
            .take(2)
            .map(|(k, v)| (*k, *v))
            .collect::<BTreeMap<_, _>>();

        checkpoints.insert(checkpoint.height, checkpoint.hash);

        let update = Update {
            txids: txs
                .into_iter()
                .map(|txid| (txid, TxHeight::Confirmed(checkpoint.height)))
                .collect(),
            checkpoints,
        };

        self.apply_update(update)
    }

    pub fn iter_txids(
        &self,
    ) -> impl DoubleEndedIterator<Item = (TxHeight, Txid)> + ExactSizeIterator + '_ {
        self.txid_by_height.iter().map(|(k, v)| (*k, *v))
    }

    pub fn iter_txids_in_height_range<R: RangeBounds<TxHeight>>(
        &self,
        range: R,
    ) -> impl DoubleEndedIterator + '_ {
        fn map_bound(bound: Bound<&TxHeight>) -> Bound<(TxHeight, Txid)> {
            match bound {
                Bound::Unbounded => Bound::Unbounded,
                Bound::Included(x) => Bound::Included((x.clone(), Txid::all_zeros())),
                Bound::Excluded(x) => Bound::Excluded((x.clone(), Txid::all_zeros())),
            }
        }

        self.txid_by_height
            .range((map_bound(range.start_bound()), map_bound(range.end_bound())))
    }
    pub fn full_txout(&self, graph: &TxGraph, outpoint: OutPoint) -> Option<FullTxOut> {
        let height = self.transaction_height(outpoint.txid)?;

        let txout = graph.txout(outpoint).cloned()?;

        let spent_by = graph
            .outspend(&outpoint)
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
            height,
            spent_by,
        })
    }

    pub fn set_checkpoint_limit(&mut self, limit: Option<usize>) {
        self.checkpoint_limit = limit;
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
    pub fn is_unspent(&self, graph: &TxGraph, outpoint: &OutPoint) -> Option<bool> {
        let txids = graph.outspend(outpoint)?;
        Some(
            txids
                .iter()
                .all(|&txid| self.transaction_height(txid).is_none()),
        )
    }
}

/// Represents an [`Update`] that could be applied to [`SparseChain`].
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// List of transactions in this checkpoint. They needs to be consistent with [`SparseChain`]'s
    /// state for the [`Update`] to be included.
    pub txids: HashMap<Txid, TxHeight>,
    /// The chain represented by checkpoints *must* connect to the existing sparse chain or to the
    /// empty chain. That it is it must have a `BlockId` that matches one of the existing
    /// checkpoints or it is connects to the empty chain. To connect to the empty chain, the
    /// existing chain must be empty OR one of the checkpoints must be at the same height as the
    /// first checkpoint in the sparse chain.
    pub checkpoints: BTreeMap<u32, BlockHash>,
}

/// Represents the set of changes as result of a successful [`Update`].
#[derive(Debug, Default, PartialEq)]
pub struct ChangeSet {
    pub checkpoints: HashMap<u32, Change<BlockHash>>,
    pub txids: HashMap<Txid, Change<TxHeight>>,
}

impl ChangeSet {
    pub fn merge(mut self, new_set: Self) -> Result<Self, MergeFailure> {
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
pub enum MergeFailure {
    Checkpoint(MergeConflict<u32, BlockHash>),
    Txid(MergeConflict<Txid, TxHeight>),
}

#[derive(Debug, Default)]
pub struct MergeConflict<K, V> {
    pub key: K,
    pub change: Change<V>,
    pub new_change: Change<V>,
}

impl Display for MergeFailure {
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
impl std::error::Error for MergeFailure {}

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
pub struct FullTxOut {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub height: TxHeight,
    pub spent_by: Option<Txid>,
}
