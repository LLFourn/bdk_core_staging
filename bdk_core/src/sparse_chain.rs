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
}

#[derive(Clone, Debug, PartialEq)]
pub enum StaleReason {
    LastValidConflictsNewTip {
        new_tip: BlockId,
        last_valid: BlockId,
    },
    UnexpectedLastValid {
        got: Option<BlockId>,
        expected: Option<BlockId>,
    },
    TxidHeightGreaterThanTip {
        new_tip: BlockId,
        txid: (Txid, Option<u32>),
    },
    TxUnexpectedlyMoved {
        txid: Txid,
        from: Option<u32>,
        to: Option<u32>,
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
            TxHeight::Unconfirmed
        } else {
            TxHeight::Confirmed(*self.txid_to_index.get(txid)?)
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

        let matching_checkpoint = self.checkpoint_at(block_id.height);
        if matches!(matching_checkpoint, Some(id) if id != block_id) {
            checkpoint.invalidate = matching_checkpoint;
        }

        self.apply_checkpoint(checkpoint)
    }

    /// Applies a new candidate checkpoint to the tracker.
    #[must_use]
    pub fn apply_checkpoint(&mut self, new_checkpoint: CheckpointCandidate) -> ApplyResult {
        // if there is no `invalidate`, `last_valid` should be the last checkpoint in sparsechain
        // if there is `invalidate`, `last_valid` should be the checkpoint preceding `invalidate`
        let expected_last_valid = {
            let upper_bound = new_checkpoint
                .invalidate
                .map(|b| b.height)
                .unwrap_or(u32::MAX);
            self.checkpoints
                .range(..upper_bound)
                .last()
                .map(|(&height, &hash)| BlockId { height, hash })
        };
        if new_checkpoint.last_valid != expected_last_valid {
            return ApplyResult::Stale(StaleReason::UnexpectedLastValid {
                got: new_checkpoint.last_valid,
                expected: expected_last_valid,
            });
        }

        // `new_tip.height` should be greater or equal to `last_valid.height`
        // if `new_tip.height` is equal to `last_valid.height`, the hashes should also be the same
        if let Some(last_valid) = expected_last_valid {
            if new_checkpoint.new_tip.height < last_valid.height
                || new_checkpoint.new_tip.height == last_valid.height
                    && new_checkpoint.new_tip.hash != last_valid.hash
            {
                return ApplyResult::Stale(StaleReason::LastValidConflictsNewTip {
                    new_tip: new_checkpoint.new_tip,
                    last_valid,
                });
            }
        }

        for (txid, tx_height) in &new_checkpoint.txids {
            // ensure new_height does not surpass latest checkpoint
            if matches!(tx_height, Some(tx_h) if tx_h > &new_checkpoint.new_tip.height) {
                return ApplyResult::Stale(StaleReason::TxidHeightGreaterThanTip {
                    new_tip: new_checkpoint.new_tip,
                    txid: (*txid, tx_height.clone()),
                });
            }

            // ensure all currently confirmed txs are still at the same height (unless, if they are
            // to be invalidated)
            if let Some(&height) = self.txid_to_index.get(txid) {
                // no need to check consistency if height will be invalidated
                // tx is consistent if height stays the same
                if matches!(new_checkpoint.invalidate, Some(invalid) if height >= invalid.height)
                    || matches!(tx_height, Some(new_height) if *new_height == height)
                {
                    continue;
                }

                // inconsistent
                return ApplyResult::Stale(StaleReason::TxUnexpectedlyMoved {
                    txid: *txid,
                    from: Some(height),
                    to: *tx_height,
                });
            }
        }

        if let Some(invalid) = &new_checkpoint.invalidate {
            self.invalidate_checkpoints(invalid.height);
        }

        // record latest checkpoint (if any)
        self.checkpoints
            .entry(new_checkpoint.new_tip.height)
            .or_insert(new_checkpoint.new_tip.hash);

        for (txid, conf) in new_checkpoint.txids {
            match conf {
                Some(height) => {
                    if self.txid_by_height.insert((height, txid)) {
                        self.txid_to_index.insert(txid, height);
                        self.mempool.remove(&txid);
                    }
                }
                None => {
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

        let txout = graph.txout(&outpoint).cloned()?;

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

/// TODO: How do we ensure `txids` do not have a height greater than `new_tip`?
/// TODO: Add `relevant_blocks: Vec<BlockId>`
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCandidate {
    /// List of transactions in this checkpoint. They needs to be consistent with tracker's state
    /// for the new checkpoint to be included.
    pub txids: Vec<(Txid, Option<u32>)>,
    /// The new checkpoint can be applied upon this tip. A tracker will usually reject updates that
    /// do not have `last_valid` equal to it's latest valid checkpoint.
    pub last_valid: Option<BlockId>,
    /// Invalidates a block before considering this checkpoint.
    pub invalidate: Option<BlockId>,
    /// Sets the tip that this checkpoint was created for. All data in this checkpoint must be valid
    /// with respect to this tip.
    pub new_tip: BlockId,
}

impl CheckpointCandidate {
    /// Helper function to create a template checkpoint candidate.
    pub fn new(last_valid: Option<BlockId>, new_tip: BlockId) -> Self {
        Self {
            txids: Vec::new(),
            last_valid,
            invalidate: None,
            new_tip,
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
