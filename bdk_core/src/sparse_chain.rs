use core::ops::RangeBounds;

use crate::{collections::*, Vec};
use crate::{BlockId, BlockTime};
use bitcoin::hashes::{Hash, HashEngine};
use bitcoin::{hashes::sha256, BlockHash, OutPoint, Transaction};
use bitcoin::{Script, Txid};

#[derive(Clone, Debug, Default)]
pub struct SparseChain {
    /// Checkpoint data indexed by height
    checkpoints: BTreeMap<u32, CheckpointData>,
    /// List of all known spends. Including our's and other's outpoints. Both confirmed and unconfirmed.
    /// This is useful to track all inputs we might ever care about.
    spends: BTreeMap<OutPoint, Txid>,
    /// A transaction store of all potentially interesting transactions. Including ones in mempool.
    txs: HashMap<Txid, TxAtBlock>,
    /// A list of mempool txids
    mempool: HashSet<Txid>,
    /// The maximum number of checkpoints that the descriptor should store. When a new checkpoint is
    /// added which would push it above the limit we merge the oldest two checkpoints together.
    checkpoint_limit: Option<usize>,
}

/// We keep this for two reasons:
///
/// 1. If we have two different checkpoints we can tell quickly if they disagree and if so at which
/// height do they disagree (set reconciliation).
/// 2. We want to be able to delete old checkpoints by merging their Txids
/// into a newer one. With this digest we can do that without changing the identity of the
/// checkpoint that has the new Txids merged into it.
#[derive(Clone, Default)]
struct CheckpointData {
    block_hash: BlockHash,
    ordered_txids: BTreeSet<(u32, Txid)>,
    txid_digest: sha256::HashEngine,
}

impl core::fmt::Debug for CheckpointData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CheckpointData")
            .field("block_hash", &self.block_hash)
            .field("txids", &self.ordered_txids)
            .field(
                "txid_digest",
                &sha256::Hash::from_engine(self.txid_digest.clone()),
            )
            .finish()
    }
}

/// The result of attempting to apply a checkpoint
#[derive(Clone, Debug, PartialEq)]
pub enum ApplyResult {
    /// The checkpoint was applied successfully.
    // TODO: return a diff
    Ok,
    /// The checkpoint cannot be applied to the current state because it does not apply to the current
    /// tip of the tracker or does not invalidate the right checkpoint such that it does.
    // TDOD: Have a stale reason
    Stale(StaleReason),
    /// The checkpoint you tried to apply was inconsistent with the current state.
    ///
    /// To forcibly apply the checkpoint you must invalidate a the block that `conflicts_with` is in (or one preceeding it).
    Inconsistent { txid: Txid, conflicts_with: Txid },
}

#[derive(Clone, Debug, PartialEq)]
pub enum StaleReason {
    InvalidationHashNotMatching {
        invalidates: BlockId,
        expected: Option<BlockHash>,
    },
    BaseTipNotMatching {
        got: Option<BlockId>,
        expected: BlockId,
    },
}

impl SparseChain {
    /// Set the checkpoint limit for this tracker.
    /// If the limit is exceeded the last two checkpoints are merged together.
    pub fn set_checkpoint_limit(&mut self, limit: usize) {
        assert!(limit > 1);
        self.checkpoint_limit = Some(limit);
        self.apply_checkpoint_limit()
    }

    pub fn outspend(&self, outpoint: OutPoint) -> Option<Txid> {
        self.spends.get(&outpoint).cloned()
    }

    /// The outputs from the transaction with id `txid` that have been spent.
    ///
    /// Each item contains the output index and the txid that spent that output.
    pub fn outspends(&self, txid: Txid) -> impl DoubleEndedIterator<Item = (u32, Txid)> + '_ {
        let start = OutPoint { txid, vout: 0 };
        let end = OutPoint {
            txid,
            vout: u32::MAX,
        };
        self.spends
            .range(start..=end)
            .map(|(outpoint, txid)| (outpoint.vout, *txid))
    }

    fn apply_checkpoint_limit(&mut self) {
        // we merge the oldest two checkpoints because they are least likely to be reverted.
        while self.checkpoints.len() > self.checkpoint_limit.unwrap_or(usize::MAX) {
            let oldest = *self.checkpoints.iter().next().unwrap().0;
            self.merge_checkpoint(oldest);
        }
    }

    /// Get the transaction ids in a particular checkpoint.
    ///
    /// The `Txid`s are ordered first by their confirmation height (ascending) and then lexically by their `Txid`.
    ///
    /// ## Panics
    ///
    /// This will panic if a checkpoint doesn't exist with `checkpoint_id`
    pub fn checkpoint_txids(
        &self,
        checkpoint_id: BlockId,
    ) -> impl DoubleEndedIterator<Item = Txid> + '_ {
        let data = self
            .checkpoints
            .get(&checkpoint_id.height)
            .expect("the tracker did not have a checkpoint at that height");
        assert_eq!(
            data.block_hash, checkpoint_id.hash,
            "tracker had a different block hash for checkpoint at that height"
        );

        data.ordered_txids.iter().map(|(_, txid)| *txid)
    }

    /// Gets the SHA256 hash of all the `Txid`s of all the transactions included in all checkpoints
    /// up to and including `checkpoint_id`.
    ///
    /// ## Panics
    ///
    /// This will panic if a checkpoint doesn't exist with `checkpoint_id`
    pub fn txid_digest_at(&self, checkpoint_id: BlockId) -> sha256::Hash {
        let data = self
            .checkpoints
            .get(&checkpoint_id.height)
            .expect("the tracker did not have a checkpoint at that height");
        assert_eq!(
            data.block_hash, checkpoint_id.hash,
            "tracker had a different block hash for checkpoint at that height"
        );

        sha256::Hash::from_engine(data.txid_digest.clone())
    }

    /// Get the BlockId for the last known tip.
    pub fn latest_checkpoint(&self) -> Option<BlockId> {
        self.checkpoints
            .iter()
            .last()
            .map(|(height, data)| BlockId {
                height: *height,
                hash: data.block_hash,
            })
    }

    /// Get the checkpoint id at the given height if it exists
    pub fn checkpoint_at(&self, height: u32) -> Option<BlockId> {
        let data = self.checkpoints.get(&height)?;
        Some(BlockId {
            height,
            hash: data.block_hash,
        })
    }

    /// Return an iterator over the checkpoint locations in a height range.
    pub fn iter_checkpoints(
        &self,
        range: impl RangeBounds<u32>,
    ) -> impl DoubleEndedIterator<Item = BlockId> + '_ {
        self.checkpoints.range(range).map(|(height, data)| BlockId {
            height: *height,
            hash: data.block_hash,
        })
    }

    /// Apply transactions that are all confirmed in a given block
    pub fn apply_block_txs(
        &mut self,
        block_id: BlockId,
        block_timestamp: u64,
        transactions: impl IntoIterator<Item = Transaction>,
    ) -> ApplyResult {
        let mut checkpoint = CheckpointCandidate {
            transactions: transactions
                .into_iter()
                .map(|tx| TxAtBlock {
                    tx,
                    confirmation_time: Some(BlockTime {
                        height: block_id.height,
                        time: block_timestamp,
                    }),
                })
                .collect(),
            base_tip: self.latest_checkpoint(),
            invalidate: None,
            new_tip: block_id,
        };
        if let Some(matching_checkpoint) = self.checkpoint_at(block_id.height) {
            if matching_checkpoint.hash != block_id.hash {
                checkpoint.invalidate = Some(matching_checkpoint);
                checkpoint.base_tip =
                    self.checkpoints
                        .range(..block_id.height)
                        .last()
                        .map(|(height, data)| BlockId {
                            height: *height,
                            hash: data.block_hash,
                        })
            }
        }

        self.apply_checkpoint(checkpoint)
    }

    /// Applies a new candidate checkpoint to the tracker.
    #[must_use]
    pub fn apply_checkpoint(&mut self, mut new_checkpoint: CheckpointCandidate) -> ApplyResult {
        new_checkpoint.transactions.retain(|tx| {
            if let Some(confirmation_time) = tx.confirmation_time {
                confirmation_time.height <= new_checkpoint.new_tip.height
            } else {
                true
            }
        });

        // we set to u32::MAX in case of None since it means no tx will be excluded from conflict checks
        let invalidation_height = new_checkpoint
            .invalidate
            .map(|bt| bt.height)
            .unwrap_or(u32::MAX);

        // Do consistency checks first so we don't mutate anything until we're sure the update is
        // valid. We check for two things
        // 1. There's no "known" transaction in the new checkpoint with same txid but different conf_time
        // 2. No transaction double spends one of our existing confirmed transactions.

        // We simply ignore transactions in the checkpoint that have a confirmation time greater
        // than the checkpoint height. I felt this was better for the caller than creating an error
        // type.

        for tx in &new_checkpoint.transactions {
            if let Some((txid, conflicts_with)) = self.check_consistency(tx, invalidation_height) {
                return ApplyResult::Inconsistent {
                    txid,
                    conflicts_with,
                };
            }
        }

        let base_tip_cmp = match new_checkpoint.invalidate {
            Some(checkpoint_reset) => self
                .checkpoints
                .range(..checkpoint_reset.height)
                .last()
                .map(|(height, data)| BlockId {
                    height: *height,
                    hash: data.block_hash,
                }),
            None => self.latest_checkpoint(),
        };

        if let Some(base_tip) = base_tip_cmp {
            if new_checkpoint.base_tip != Some(base_tip) {
                return ApplyResult::Stale(StaleReason::BaseTipNotMatching {
                    got: new_checkpoint.base_tip,
                    expected: base_tip,
                });
            }
        }

        if let Some(checkpoint_reset) = new_checkpoint.invalidate {
            match self.checkpoints.get(&checkpoint_reset.height) {
                Some(existing_checkpoint) => {
                    if existing_checkpoint.block_hash == checkpoint_reset.hash {
                        self.invalidate_checkpoint(checkpoint_reset.height);
                    } else {
                        return ApplyResult::Stale(StaleReason::InvalidationHashNotMatching {
                            invalidates: checkpoint_reset,
                            expected: Some(existing_checkpoint.block_hash),
                        });
                    }
                }
                None => {
                    return ApplyResult::Stale(StaleReason::InvalidationHashNotMatching {
                        invalidates: checkpoint_reset,
                        expected: None,
                    })
                }
            }
        }

        // If the current tip is empty (i.e. just tracking lastest height) we can just remove it.
        if let Some(latest_checkpoint) = self.latest_checkpoint() {
            if self.checkpoint_txids(latest_checkpoint).next().is_none() {
                self.checkpoints.remove(&latest_checkpoint.height);
            }
        }

        self.checkpoints
            .entry(new_checkpoint.new_tip.height)
            .or_insert_with(|| CheckpointData {
                block_hash: new_checkpoint.new_tip.hash,
                ..Default::default()
            });

        let mut deepest_change = None;
        for tx in new_checkpoint.transactions {
            if let Some(change) = self.add_tx(tx) {
                deepest_change = Some(deepest_change.unwrap_or(u32::MAX).min(change));
            }
        }
        if let Some(change_depth) = deepest_change {
            self.recompute_txid_digests(change_depth);
        }

        self.apply_checkpoint_limit();

        debug_assert!(self.is_latest_checkpoint_hash_correct());

        ApplyResult::Ok
    }

    fn check_consistency(
        &self,
        TxAtBlock {
            tx,
            confirmation_time,
        }: &TxAtBlock,
        invalidation_height: u32,
    ) -> Option<(Txid, Txid)> {
        let txid = tx.txid();
        if let Some(existing) = self.txs.get(&tx.txid()) {
            if let Some(existing_time) = existing.confirmation_time {
                if existing_time.height >= invalidation_height {
                    // no need to consider conflicts for txs that are about to be invalidated
                    return None;
                }
                if confirmation_time != &Some(existing_time) {
                    if existing_time.height < invalidation_height {
                        return Some((txid, existing.tx.txid()));
                    }
                }
            }
        }

        for input in &tx.input {
            let prevout = &input.previous_output;
            if let Some(spent_from) = self.txs.get(&prevout.txid) {
                if spent_from.tx.output.len() as u32 <= prevout.vout {
                    // an input is spending from a tx we know about but doesn't have that output
                    return Some((tx.txid(), prevout.txid));
                }
            }
        }

        let conflicts = tx
            .input
            .iter()
            .filter_map(|input| self.spends.get(&input.previous_output))
            .filter(|conflict_txid| **conflict_txid != tx.txid());

        for conflicting_txid in conflicts {
            if let Some(conflicting_conftime) = self
                .txs
                .get(conflicting_txid)
                .expect("must exist")
                .confirmation_time
            {
                // no need to consider conflicts for txs that are about to be invalidated
                if conflicting_conftime.height >= invalidation_height {
                    continue;
                }

                return Some((txid, *conflicting_txid));
            }
        }
        None
    }

    /// Performs recomputation of transaction digest of checkpoint data
    /// from the given height.
    fn recompute_txid_digests(&mut self, from: u32) {
        let mut prev_accum_digest = self
            .checkpoints
            .range(..from)
            .last()
            .map(|(_, prev)| prev.txid_digest.clone())
            .unwrap_or_else(sha256::HashEngine::default);

        for (_height, data) in self.checkpoints.range_mut(from..) {
            let mut txid_digest = prev_accum_digest.clone();
            for (_, txid) in &data.ordered_txids {
                txid_digest.input(txid);
            }

            data.txid_digest = txid_digest.clone();
            prev_accum_digest = txid_digest;
        }
    }

    /// Takes the checkpoint at a height and merges its transactions into the next checkpoint
    pub fn merge_checkpoint(&mut self, height: u32) {
        if let Some(checkpoint) = self.checkpoints.remove(&height) {
            match self.checkpoints.range_mut((height + 1)..).next() {
                Some((_, next_one)) => {
                    next_one.ordered_txids.extend(checkpoint.ordered_txids);
                }
                None => {
                    // put it back because there's no checkpoint greater than it
                    self.checkpoints.insert(height, checkpoint);
                }
            }
        }
        debug_assert!(self.is_latest_checkpoint_hash_correct());
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) {
        let mempool = core::mem::replace(&mut self.mempool, Default::default());
        for txid in mempool {
            self.remove_tx(txid);
        }
        debug_assert!(self.mempool.is_empty())
    }

    /// Reverse everything of the Block with given hash and height.
    pub fn disconnect_block(&mut self, block_id: BlockId) {
        if let Some(checkpoint_data) = self.checkpoints.get(&block_id.height) {
            if checkpoint_data.block_hash == block_id.hash {
                // Can't guarantee that mempool is consistent with chain after we disconnect a block so we
                // clear it.
                self.clear_mempool();
                self.invalidate_checkpoint(block_id.height);
            }
        }
    }

    /// Remove a transaction from internal store.
    // This can only be called when a checkpoint is removed.
    fn remove_tx(&mut self, txid: Txid) {
        let aug_tx = match self.txs.remove(&txid) {
            Some(aug_tx) => aug_tx,
            None => {
                debug_assert!(!self.mempool.contains(&txid), "Consistency check");
                return;
            }
        };

        for input in &aug_tx.tx.input {
            if let Some(tx_that_spends) = self.spends.remove(&input.previous_output) {
                debug_assert_eq!(
                    tx_that_spends, txid,
                    "the one that spent it must be this one"
                );
            }
        }

        self.mempool.remove(&txid);
    }

    // Returns the checkpoint height at which it got added so we can recompute the txid digest from
    // that point.
    fn add_tx(
        &mut self,
        TxAtBlock {
            tx,
            confirmation_time,
        }: TxAtBlock,
    ) -> Option<u32> {
        let txid = tx.txid();

        // Look for conflicts to determine whether we should add this transaction or remove the one
        // it conflicts with. Note that the txids in conflicts will always be unconfirmed
        // transactions (dealing with confirmed conflicts is done outside and is usually an error).
        let conflicts = tx
            .input
            .iter()
            .filter_map(|input| Some(*self.spends.get(&input.previous_output)?))
            .collect::<Vec<_>>();

        if confirmation_time.is_some() {
            // Because we made sure we only have mempool transactions in conflict list, if this one
            // is already confirmed, its safe to remove them.
            for conflicting_txid in conflicts {
                self.remove_tx(conflicting_txid);
            }
        } else {
            // in this branch we have a mempool confict
            // TODO: add some way to customize the way conflicts are resolved in mempool
            for conflicting_txid in conflicts {
                self.remove_tx(conflicting_txid);
            }
        }

        for input in tx.input.iter() {
            if input.previous_output.is_null() {
                continue;
            }
            let removed = self.spends.insert(input.previous_output, txid);
            debug_assert_eq!(
                removed, None,
                "we should have already removed all conflicts!"
            );
        }

        self.txs.insert(
            txid,
            TxAtBlock {
                tx,
                confirmation_time,
            },
        );

        match confirmation_time {
            Some(confirmation_time) => {
                // Find the first checkpoint above or equal to the tx's height
                let (checkpoint_height, checkpoint_data) = self
                    .checkpoints
                    .range_mut(confirmation_time.height..)
                    .next()
                    .expect("the caller must have checked that no txs are outside of range");

                checkpoint_data
                    .ordered_txids
                    .insert((confirmation_time.height, txid));

                Some(*checkpoint_height)
            }
            None => {
                self.mempool.insert(txid);
                None
            }
        }
    }

    // Invalidate all checkpoints after the given height
    fn invalidate_checkpoint(&mut self, height: u32) {
        let removed = self.checkpoints.split_off(&height);

        for (_height, data) in removed {
            for (_, txid) in data.ordered_txids {
                self.remove_tx(txid);
            }
        }
    }

    /// Iterate over all transactions in our transaction store.
    /// Can be both related/unrelated and/or confirmed/unconfirmed.
    pub fn iter_tx(&self) -> impl Iterator<Item = (Txid, &TxAtBlock)> {
        self.txs.iter().map(|(txid, tx)| (*txid, tx))
    }

    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut> {
        let tx_in_block = self.txs.get(&outpoint.txid)?;
        let spent_by = self.outspend(outpoint);
        let txout = tx_in_block.tx.output.get(outpoint.vout as usize)?;

        Some(FullTxOut {
            value: txout.value,
            spent_by,
            outpoint,
            script_pubkey: txout.script_pubkey.clone(),
            confirmed_at: tx_in_block.confirmation_time,
        })
    }

    /// Iterates over all transactions related to the descriptor ordered by decending confirmation
    /// with those transactions that are unconfirmed first.
    ///
    /// "related" means that the transactoin has an output with a script pubkey produced by the
    /// descriptor or it spends from such an output.
    // TODO:  maybe change this so we can iterate over a height range using checkpoints
    pub fn iter_tx_by_confirmation_time(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Txid, &TxAtBlock)> + '_ {
        // Since HashSet is not necessarily a DoubleEndedIterator we collect into a vector first.
        let mempool_tx = self
            .mempool
            .iter()
            .map(|txid| (*txid, self.txs.get(txid).unwrap()))
            .collect::<Vec<_>>();
        let confirmed_tx = self.checkpoints.iter().rev().flat_map(|(_, data)| {
            data.ordered_txids
                .iter()
                .map(|(_, txid)| (*txid, self.txs.get(txid).unwrap()))
        });

        mempool_tx.into_iter().chain(confirmed_tx)
    }

    pub fn unconfirmed(&self) -> &HashSet<Txid> {
        &self.mempool
    }

    /// Get a stored transaction given its `Txid`.
    pub fn get_tx(&self, txid: Txid) -> Option<&TxAtBlock> {
        self.txs.get(&txid)
    }

    /// internal debug function to double check correctness of the accumulated digest at the tip
    #[must_use]
    fn is_latest_checkpoint_hash_correct(&self) -> bool {
        if let Some(tip) = self.latest_checkpoint() {
            let mut txs = self
                .iter_tx()
                .filter(|(_, tx)| tx.confirmation_time.is_some())
                .collect::<Vec<_>>();
            txs.sort_by_key(|(_, tx)| (tx.confirmation_time.unwrap().height, tx.tx.txid()));
            let mut hasher = sha256::HashEngine::default();
            for (txid, _) in txs {
                hasher.input(&txid);
            }
            let txid_hash = sha256::Hash::from_engine(hasher);
            self.txid_digest_at(tip) == txid_hash
        } else {
            true
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCandidate {
    /// List of transactions in this checkpoint. They needs to be consistent with tracker's state
    /// for the new checkpoint to be included.
    pub transactions: Vec<TxAtBlock>,
    /// The new checkpoint can be applied upon this tip. A tracker will usually reject updates that
    /// do not have `base_tip` equal to it's latest valid checkpoint.
    pub base_tip: Option<BlockId>,
    /// Invalidates a checkpoint before considering this checkpoint.
    pub invalidate: Option<BlockId>,
    /// Sets the tip that this checkpoint was creaed for. All data in this checkpoint must be valid
    /// with respect to this tip.
    pub new_tip: BlockId,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TxAtBlock {
    pub tx: Transaction,
    pub confirmation_time: Option<BlockTime>,
}

/// A `TxOut` with as much data as we can retreive about it
#[derive(Debug, Clone, PartialEq)]
pub struct FullTxOut {
    pub value: u64,
    pub spent_by: Option<Txid>,
    pub outpoint: OutPoint,
    pub confirmed_at: Option<BlockTime>,
    pub script_pubkey: Script,
}
