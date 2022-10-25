use core::{fmt::Display, ops::RangeBounds};

use crate::{alloc::string::String, collections::*, BlockId, TxGraph, Vec};
use bitcoin::{hashes::Hash, BlockHash, OutPoint, TxOut, Txid};

#[derive(Clone, Debug, Default)]
pub struct SparseChain {
    /// Block height to checkpoint data.
    checkpoints: BTreeMap<u32, BlockHash>,
    /// Txids prepended by confirmation height.
    txid_by_height: BTreeSet<(u32, Txid)>,
    /// Confirmation heights of txids.
    txid_to_index: HashMap<Txid, u32>,
    /// A list of mempool txids.
    mempool: HashSet<Txid>,
    /// Limit number of checkpoints.
    checkpoint_limit: Option<usize>,
}

/// Represents an update failure of [`SparseChain`].
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure {
    /// The [`Update`] is total bogus. Cannot be applied to any [`SparseChain`].
    Bogus(BogusReason),

    /// The [`Update`] cannot be applied to this [`SparseChain`] because the `last_valid` value does
    /// not match with the current state of the chain.
    Stale {
        got_last_valid: Option<BlockId>,
        expected_last_valid: Option<BlockId>,
    },

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
    /// `last_valid` conflicts with `new_tip`.
    LastValidConflictsNewTip {
        new_tip: BlockId,
        last_valid: BlockId,
    },

    /// At least one `txid` has a confirmation height greater than `new_tip`.
    TxHeightGreaterThanTip {
        new_tip: BlockId,
        tx: (Txid, TxHeight),
    },
}

impl core::fmt::Display for UpdateFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        fn print_block(id: &BlockId) -> String {
            format!("{} @ {}", id.hash, id.height)
        }

        fn print_block_opt(id: &Option<BlockId>) -> String {
            match id {
                Some(id) => print_block(id),
                None => "None".into(),
            }
        }

        match self {
            Self::Bogus(reason) => {
                write!(f, "bogus update: ")?;
                match reason {
                    BogusReason::LastValidConflictsNewTip { new_tip, last_valid } =>
                        write!(f, "last_valid ({}) conflicts new_tip ({})", 
                            print_block(last_valid), print_block(new_tip)),

                    BogusReason::TxHeightGreaterThanTip { new_tip, tx: txid } =>
                        write!(f, "tx ({}) confirmation height ({}) is greater than new_tip ({})", 
                            txid.0, txid.1, print_block(new_tip)),
                }
            },
            Self::Stale { got_last_valid, expected_last_valid } =>
                write!(f, "stale update: got last_valid ({}) when expecting ({})", 
                    print_block_opt(got_last_valid), print_block_opt(expected_last_valid)),

            Self::Inconsistent { inconsistent_txid, original_height, update_height } =>
                write!(f, "inconsistent update: first inconsistent tx is ({}) which had confirmation height ({}), but is ({}) in the update", 
                    inconsistent_txid, original_height, update_height),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for UpdateFailure {}

impl SparseChain {
    /// Get the transaction ids in a particular checkpoint.
    ///
    /// The `Txid`s are ordered first by their confirmation height (ascending) and then lexically by their `Txid`.
    ///
    /// ## Panics
    ///
    /// This will panic if a checkpoint doesn't exist with `checkpoint_id`
    pub fn checkpoint_txids(
        &self,
        block_id: BlockId,
    ) -> impl DoubleEndedIterator<Item = &(u32, Txid)> + '_ {
        let block_hash = self
            .checkpoints
            .get(&block_id.height)
            .expect("the tracker did not have a checkpoint at that height");
        assert_eq!(
            block_hash, &block_id.hash,
            "tracker had a different block hash for checkpoint at that height"
        );

        let h = block_id.height;

        self.txid_by_height.range((h, Txid::all_zeros())..)
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

    /// Return height of tx (if any).
    pub fn transaction_height(&self, txid: Txid) -> Option<TxHeight> {
        Some(if self.mempool.contains(&txid) {
            TxHeight::Unconfirmed
        } else {
            TxHeight::Confirmed(*self.txid_to_index.get(&txid)?)
        })
    }

    /// Return an iterator over the checkpoint locations in a height range.
    pub fn iter_checkpoints(
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
                .mempool
                .iter()
                .map(|txid| (*txid, Change::new_insertion(TxHeight::Unconfirmed)))
                .chain(self.txid_by_height.iter().map(|(height, txid)| {
                    (*txid, Change::new_insertion(TxHeight::Confirmed(*height)))
                }))
                .collect(),
        }
    }

    /// Apply transactions that are all confirmed in a given block
    pub fn apply_block_txs(
        &mut self,
        block_id: BlockId,
        transactions: impl IntoIterator<Item = Txid>,
    ) -> Result<ChangeSet, UpdateFailure> {
        let mut checkpoint = Update {
            txids: transactions
                .into_iter()
                .map(|txid| (txid, TxHeight::Confirmed(block_id.height)))
                .collect(),
            last_valid: self.latest_checkpoint(),
            invalidate: None,
            new_tip: block_id,
        };

        let matching_checkpoint = self.checkpoint_at(block_id.height);
        if matches!(matching_checkpoint, Some(id) if id != block_id) {
            checkpoint.invalidate = matching_checkpoint;
        }

        self.apply_update(checkpoint)
    }

    /// Applies a new [`Update`] to the tracker.
    #[must_use]
    pub fn apply_update(&mut self, update: Update) -> Result<ChangeSet, UpdateFailure> {
        // if there is no `invalidate`, `last_valid` should be the last checkpoint in sparsechain
        // if there is `invalidate`, `last_valid` should be the checkpoint preceding `invalidate`
        let expected_last_valid = {
            let upper_bound = update.invalidate.map(|b| b.height).unwrap_or(u32::MAX);
            self.checkpoints
                .range(..upper_bound)
                .last()
                .map(|(&height, &hash)| BlockId { height, hash })
        };
        if update.last_valid != expected_last_valid {
            return Result::Err(UpdateFailure::Stale {
                got_last_valid: update.last_valid,
                expected_last_valid: expected_last_valid,
            });
        }

        // `new_tip.height` should be greater or equal to `last_valid.height`
        // if `new_tip.height` is equal to `last_valid.height`, the hashes should also be the same
        if let Some(last_valid) = expected_last_valid {
            if update.new_tip.height < last_valid.height
                || update.new_tip.height == last_valid.height
                    && update.new_tip.hash != last_valid.hash
            {
                return Result::Err(UpdateFailure::Bogus(
                    BogusReason::LastValidConflictsNewTip {
                        new_tip: update.new_tip,
                        last_valid,
                    },
                ));
            }
        }

        for (txid, tx_height) in &update.txids {
            // ensure new_height does not surpass latest checkpoint
            if matches!(tx_height, TxHeight::Confirmed(tx_h) if tx_h > &update.new_tip.height) {
                return Result::Err(UpdateFailure::Bogus(BogusReason::TxHeightGreaterThanTip {
                    new_tip: update.new_tip,
                    tx: (*txid, tx_height.clone()),
                }));
            }

            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated)
            if let Some(&height) = self.txid_to_index.get(txid) {
                // no need to check consistency if height will be invalidated
                if matches!(update.invalidate, Some(invalid) if height >= invalid.height)
                    // tx is consistent if height stays the same
                    || matches!(tx_height, TxHeight::Confirmed(new_height) if *new_height == height)
                {
                    continue;
                }

                // inconsistent
                return Result::Err(UpdateFailure::Inconsistent {
                    inconsistent_txid: *txid,
                    original_height: TxHeight::Confirmed(height),
                    update_height: *tx_height,
                });
            }
        }

        // obtain initial change_set by invalidating checkpoints (if needed)
        let mut change_set = update
            .invalidate
            .map(|invalid| self.invalidate_checkpoints(invalid.height))
            .unwrap_or_default();

        // record latest checkpoint (if any)
        if !self.checkpoints.contains_key(&update.new_tip.height) {
            self.checkpoints
                .insert(update.new_tip.height, update.new_tip.hash);

            change_set
                .checkpoints
                .entry(update.new_tip.height)
                .and_modify(|change| change.to = Some(update.new_tip.hash))
                .or_insert_with(|| Change::new_insertion(update.new_tip.hash));
        }

        for (txid, new_conf) in update.txids {
            let original_conf = self
                .txid_to_index
                .get(&txid)
                .map(|&h| TxHeight::Confirmed(h))
                .or(self.mempool.get(&txid).map(|_| TxHeight::Unconfirmed));

            match new_conf {
                TxHeight::Confirmed(height) => {
                    if self.txid_by_height.insert((height, txid)) {
                        self.txid_to_index.insert(txid, height);
                        self.mempool.remove(&txid);

                        change_set
                            .txids
                            .entry(txid)
                            .and_modify(|change| change.to = Some(TxHeight::Confirmed(height)))
                            .or_insert_with(|| {
                                Change::new(original_conf, Some(TxHeight::Confirmed(height)))
                            });
                    }
                }
                TxHeight::Unconfirmed => {
                    if self.mempool.insert(txid) {
                        change_set
                            .txids
                            .entry(txid)
                            .and_modify(|change| change.to = Some(TxHeight::Unconfirmed))
                            .or_insert_with(|| {
                                Change::new(original_conf, Some(TxHeight::Unconfirmed))
                            });
                    }
                }
            }
        }

        if let Some(removed_checkpoints) = self.prune_checkpoints() {
            let changes = ChangeSet {
                checkpoints: removed_checkpoints
                    .into_iter()
                    .map(|(height, hash)| (height, Change::new_removal(hash)))
                    .collect(),
                ..Default::default()
            };

            change_set = change_set.merge(changes).expect("should succeed");
        }

        Result::Ok(change_set)
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) -> ChangeSet {
        let txids = self
            .mempool
            .iter()
            .map(|&txid| (txid, Change::new_removal(TxHeight::Unconfirmed)))
            .collect();
        self.mempool.clear();
        ChangeSet {
            txids,
            ..Default::default()
        }
    }

    /// Reverse everything of the Block with given hash and height.
    pub fn disconnect_block(&mut self, block_id: BlockId) -> Option<ChangeSet> {
        if let Some(checkpoint_hash) = self.checkpoints.get(&block_id.height) {
            if checkpoint_hash == &block_id.hash {
                return Some(self.invalidate_checkpoints(block_id.height));
            }
        }
        None
    }

    // Invalidate all checkpoints from the given height
    fn invalidate_checkpoints(&mut self, height: u32) -> ChangeSet {
        let checkpoints = self
            .checkpoints
            .split_off(&height)
            .into_iter()
            .map(|(height, hash)| (height, Change::new_removal(hash)))
            .collect::<HashMap<_, _>>();

        // remove both confirmed and mempool txids
        let txids = self
            .txid_by_height
            .split_off(&(height, Txid::all_zeros()))
            .into_iter()
            .map(|(h, txid)| (txid, TxHeight::Confirmed(h)))
            // chain removed mempool txids
            .chain(
                self.mempool
                    .iter()
                    .map(|&txid| (txid, TxHeight::Unconfirmed)),
            )
            // clear txid back references
            .inspect(|(txid, expected_height)| {
                let original_height = self.txid_to_index.remove(txid);
                debug_assert_eq!(*expected_height, original_height.into());
            })
            .map(|(txid, height)| (txid, Change::new_removal(height)))
            .collect::<HashMap<_, _>>();

        self.mempool.clear();
        ChangeSet { checkpoints, txids }
    }

    /// Iterates over confirmed txids, in increasing confirmations.
    pub fn iter_confirmed_txids(&self) -> impl Iterator<Item = &(u32, Txid)> + DoubleEndedIterator {
        self.txid_by_height.iter().rev()
    }

    /// Iterates over unconfirmed txids.
    pub fn iter_mempool_txids(&self) -> impl Iterator<Item = &Txid> {
        self.mempool.iter()
    }

    pub fn iter_txids(&self) -> impl Iterator<Item = (Option<u32>, Txid)> + '_ {
        let mempool_iter = self.iter_mempool_txids().map(|&txid| (None, txid));
        let confirmed_iter = self
            .iter_confirmed_txids()
            .map(|&(h, txid)| (Some(h), txid));
        mempool_iter.chain(confirmed_iter)
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
}

/// Represents an [`Update`] that could be applied to [`SparseChain`].
#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    /// List of transactions in this checkpoint. They needs to be consistent with [`SparseChain`]'s
    /// state for the [`Update`] to be included.
    pub txids: HashMap<Txid, TxHeight>,

    /// This should be the latest valid checkpoint of [`SparseChain`]; used to avoid conflicts.
    /// If `invalidate == None`, then this would be be the latest checkpoint of [`SparseChain`].
    /// If `invalidate == Some`, then this would be the checkpoint directly preceding `invalidate`.
    /// If [`SparseChain`] is empty, `last_valid` should be `None`.
    pub last_valid: Option<BlockId>,

    /// Invalidates all checkpoints from this checkpoint (inclusive).
    pub invalidate: Option<BlockId>,

    /// The latest tip that this [`Update`] is aware of. Introduced transactions cannot surpass this
    /// tip.
    pub new_tip: BlockId,
}

impl Update {
    /// Helper function to create a template update.
    pub fn new(last_valid: Option<BlockId>, new_tip: BlockId) -> Self {
        Self {
            txids: HashMap::new(),
            last_valid,
            invalidate: None,
            new_tip,
        }
    }
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

#[derive(Debug, Default, Clone, PartialEq)]
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
