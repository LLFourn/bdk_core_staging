use core::{fmt::Display, ops::RangeBounds};

use crate::{collections::*, BlockId, TxGraph, Vec};
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

/// The result of attempting to apply a checkpoint
#[derive(Clone, Debug, PartialEq)]
pub enum ApplyResult {
    /// The checkpoint was applied successfully.
    // TODO: return a diff
    Ok,
    /// The checkpoint cannot be applied to the current state because it does not apply to the current
    /// tip of the tracker, or does not invalidate the right checkpoint, or the candidate is invalid.
    Stale(StaleReason),
    /// The checkpoint you tried to apply was inconsistent with the current state.
    ///
    /// To forcibly apply the checkpoint you must invalidate a the block that `conflicts_with` is in (or one preceeding it).
    Inconsistent { txid: Txid, conflicts_with: Txid },
}

#[derive(Clone, Debug, PartialEq)]
pub enum StaleReason {
    InvalidationHashNotMatching {
        got: Option<BlockHash>,
        expected: BlockId,
    },
    LastValidHashNotMatching {
        got: Option<BlockHash>,
        expected: BlockId,
    },
    TxidHeightGreaterThanNewTip {
        tip: BlockId,
        txid: (Txid, Option<u32>),
    },
    LastValidConflictsWithInvalidate {
        last_valid: BlockId,
        invalidate: BlockId,
    },
}

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
    pub fn transaction_height(&self, txid: &Txid) -> Option<TxHeight> {
        Some(if self.mempool.contains(txid) {
            TxHeight::Mempool
        } else {
            TxHeight::Height(*self.txid_to_index.get(txid)?)
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

    /// Apply transactions that are all confirmed in a given block
    pub fn apply_block_txs(
        &mut self,
        block_id: BlockId,
        transactions: impl IntoIterator<Item = Txid>,
    ) -> ApplyResult {
        let mut checkpoint = CheckpointCandidate {
            txids: transactions
                .into_iter()
                .map(|txid| (txid, Some(block_id.height)))
                .collect(),
            last_valid: self.latest_checkpoint(),
            invalidate: None,
            new_tip: block_id,
        };

        if let Some(matching_checkpoint) = self.checkpoint_at(block_id.height) {
            if matching_checkpoint.hash != block_id.hash {
                checkpoint.invalidate = Some(matching_checkpoint);
            }
        }

        self.apply_checkpoint(checkpoint)
    }

    /// Applies a new candidate checkpoint to the tracker.
    #[must_use]
    pub fn apply_checkpoint(&mut self, new_checkpoint: CheckpointCandidate) -> ApplyResult {
        // enforce base-tip rule (if any)
        if let Some(last_valid) = &new_checkpoint.last_valid {
            // ensure last_valid & invalidate makes sense
            if let Some(invalid) = &new_checkpoint.invalidate {
                if invalid.height <= last_valid.height {
                    return ApplyResult::Stale(StaleReason::LastValidConflictsWithInvalidate {
                        last_valid: last_valid.clone(),
                        invalidate: invalid.clone(),
                    });
                }
            }

            let current_hash = self.checkpoints.get(&last_valid.height);

            // if block exists, but is of a different hash, this is invalid
            if matches!(current_hash, Some(h) if h != &last_valid.hash) {
                return ApplyResult::Stale(StaleReason::LastValidHashNotMatching {
                    got: current_hash.cloned(),
                    expected: last_valid.clone(),
                });
            }
        }

        for (txid, new_height) in &new_checkpoint.txids {
            // ensure new_height does not surpass new_tip
            if matches!(new_height, Some(h) if h > &new_checkpoint.new_tip.height) {
                return ApplyResult::Stale(StaleReason::TxidHeightGreaterThanNewTip {
                    tip: new_checkpoint.new_tip,
                    txid: (*txid, new_height.clone()),
                });
            }

            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated)
            if let Some(&height) = self.txid_to_index.get(txid) {
                // no need to check consistency if height will be invalidated
                if matches!(new_checkpoint.invalidate, Some(invalid) if height >= invalid.height) {
                    continue;
                }
                // consistent if height stays the same
                if matches!(new_height, Some(new_height) if *new_height == height) {
                    continue;
                }
                // inconsistent
                return ApplyResult::Inconsistent {
                    txid: *txid,
                    conflicts_with: *txid,
                };
            }
        }

        if let Some(invalid) = &new_checkpoint.invalidate {
            let block_hash = self.checkpoints.get(&invalid.height);
            if !matches!(block_hash, Some(h) if h == &invalid.hash) {
                return ApplyResult::Stale(StaleReason::InvalidationHashNotMatching {
                    got: block_hash.cloned(),
                    expected: invalid.clone(),
                });
            }

            self.invalidate_checkpoints(invalid.height);
        }

        self.checkpoints
            .entry(new_checkpoint.new_tip.height)
            .or_insert_with(|| new_checkpoint.new_tip.hash);

        for (txid, conf) in new_checkpoint.txids {
            match conf {
                Some(height) => {
                    if self.txid_by_height.insert((height, txid)) {
                        self.txid_to_index.insert(txid, height);
                        self.mempool.remove(&txid);
                    }
                }
                None => {
                    // TODO: Use u32::MAX for mempool?
                    self.mempool.insert(txid);
                }
            }
        }

        self.prune_checkpoints();
        ApplyResult::Ok
    }

    /// Clear the mempool list. Use with caution.
    pub fn clear_mempool(&mut self) {
        self.mempool.clear()
    }

    /// Reverse everything of the Block with given hash and height.
    pub fn disconnect_block(&mut self, block_id: BlockId) {
        if let Some(checkpoint_hash) = self.checkpoints.get(&block_id.height) {
            if checkpoint_hash == &block_id.hash {
                // Can't guarantee that mempool is consistent with chain after we disconnect a block so we
                // clear it.
                self.invalidate_checkpoints(block_id.height);
                self.clear_mempool();
            }
        }
    }

    // Invalidate all checkpoints from the given height
    fn invalidate_checkpoints(&mut self, height: u32) {
        let _removed_checkpoints = self.checkpoints.split_off(&height);
        let removed_txids = self.txid_by_height.split_off(&(height, Txid::all_zeros()));

        for (exp_h, txid) in &removed_txids {
            let h = self.txid_to_index.remove(txid);
            debug_assert!(matches!(h, Some(h) if h == *exp_h));
        }

        // TODO: have a method to make mempool consistent
        if !removed_txids.is_empty() {
            self.mempool.clear()
        }
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
        let height = self.transaction_height(&outpoint.txid)?;

        let txout = graph
            .tx(&outpoint.txid)?
            .output(outpoint.vout as _)
            .cloned()?;

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

    pub fn prune_checkpoints(&mut self) -> Option<BTreeMap<u32, BlockHash>> {
        let height = *self.checkpoints.keys().rev().nth(self.checkpoint_limit?)?;
        return Some(self.checkpoints.split_off(&height));
    }
}

/// TODO: How do we ensure `txids` do not have a height greater than `new_tip`?
/// TODO: Add `relevant_blocks: Vec<BlockId>`
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCandidate {
    /// List of transactions in this checkpoint. They needs to be consistent with tracker's state
    /// for the new checkpoint to be included.
    pub txids: Vec<(Txid, Option<u32>)>,
    /// The new checkpoint can be applied upon this tip. A tracker will usually reject updates that
    /// do not have `base_tip` equal to it's latest valid checkpoint.
    pub last_valid: Option<BlockId>,
    /// Invalidates a block before considering this checkpoint.
    pub invalidate: Option<BlockId>,
    /// Sets the tip that this checkpoint was creaed for. All data in this checkpoint must be valid
    /// with respect to this tip.
    pub new_tip: BlockId,
}

/// Represents the height in which a transaction is confirmed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TxHeight {
    Height(u32),
    Mempool,
}

impl Display for TxHeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Height(h) => core::write!(f, "confirmed_at({})", h),
            Self::Mempool => core::write!(f, "unconfirmed"),
        }
    }
}

impl TxHeight {
    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Height(_))
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
