use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use crate::{BlockId, BlockTime, PrevOuts, Vec};
use bitcoin::{
    self,
    hashes::{sha256, Hash, HashEngine},
    psbt::{self},
    BlockHash, OutPoint, Script, Transaction, Txid,
};
use miniscript::{descriptor::DefiniteDescriptorKey, Descriptor};

#[derive(Clone, Debug)]
pub struct ScriptTracker {
    /// Checkpoint data (txids)
    checkpoints: BTreeMap<u32, CheckpointData>,
    /// Index the Outpoints owned by this tracker to the derivation index of script pubkey.
    txouts: BTreeMap<OutPoint, u32>,
    /// List of all known spends. Including our's and other's outpoints. Both confirmed and unconfirmed.
    /// This is useful to track all inputs we might ever care about.
    spends: BTreeMap<OutPoint, (u32, Txid)>,
    /// Set of our unspent outpoints.
    unspent: HashSet<OutPoint>,
    /// Derived script_pubkeys ordered by derivation index.
    scripts: Vec<Script>,
    /// A reverse lookup from out script_pubkeys to derivation index
    script_indexes: HashMap<Script, u32>,
    /// A lookup from script pubkey derivation index to related outpoints
    script_txouts: BTreeMap<u32, HashSet<OutPoint>>,
    /// A set of unused derivation indices.
    unused: BTreeSet<u32>,
    /// A transaction store of all potentially interesting transactions. Including ones in mempool.
    txs: HashMap<Txid, AugmentedTx>,
    /// A list of mempool [Txid]s
    mempool: HashSet<Txid>,
    /// The maximum number of checkpoints that the descriptor should store. When a new checkpoint is
    /// added which would push it above the limit we merge the oldest two checkpoints together.
    checkpoint_limit: usize,
    /// The last tip the tracker has seen. Useful since not every checkpoint applied actually
    /// creates a checkpoint (we don't create empty checkpoints).
    last_tip_seen: Option<BlockId>,
}

/// We keep this for two reasons:
///
/// 1. If we have two different checkpoints that claims to follow the same descriptor we
/// can tell quickly if they disagree and if so at which height do they disagree.
/// 2. We want to be able to delete old checkpoints by merging their Txids into a newer one.
/// With this digest we can do that without changing the identity of the checkpoint that has
/// the new Txids merged into it.
#[derive(Clone, Default)]
struct CheckpointData {
    block_hash: BlockHash,
    ordered_txids: BTreeSet<(u32, Txid)>,
    accum_digest: sha256::HashEngine,
}

impl core::fmt::Debug for CheckpointData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CheckpointData")
            .field("block_hash", &self.block_hash)
            .field("txids", &self.ordered_txids)
            .field(
                "accum_digest",
                &sha256::Hash::from_engine(self.accum_digest.clone()),
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
    Stale,
    /// The checkpoint you tried to apply was inconsistent with the current state.
    ///
    /// To forcibly apply the checkpoint you must invalidate the checkpoint `at_checkpoint` and
    /// reapply it.
    Inconsistent {
        txid: Txid,
        conflicts_with: Txid,
        at_checkpoint: BlockId,
    },
}

impl Default for ScriptTracker {
    fn default() -> Self {
        Self {
            checkpoints: Default::default(),
            txouts: Default::default(),
            spends: Default::default(),
            unspent: Default::default(),
            scripts: Default::default(),
            script_indexes: Default::default(),
            script_txouts: Default::default(),
            unused: Default::default(),
            txs: Default::default(),
            mempool: Default::default(),
            last_tip_seen: Default::default(),
            checkpoint_limit: usize::MAX,
        }
    }
}

impl ScriptTracker {
    /// Set the checkpoint limit for this tracker.
    /// If the limit is exceeded the last two checkpoints are merged together.
    pub fn set_checkpoint_limit(&mut self, limit: usize) {
        assert!(limit > 0);
        self.checkpoint_limit = limit;
        self.apply_checkpoint_limit()
    }

    fn apply_checkpoint_limit(&mut self) {
        // we merge the oldest two checkpoints because they are least likely to be reverted.
        while self.checkpoints.len() > self.checkpoint_limit {
            let oldest = *self.checkpoints.iter().next().unwrap().0;
            self.merge_checkpoint(oldest);
        }
    }

    /// Gets the last
    pub fn last_tip_seen(&self) -> Option<BlockId> {
        self.last_tip_seen
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
    pub fn accum_digest_at(&self, checkpoint_id: BlockId) -> sha256::Hash {
        let data = self
            .checkpoints
            .get(&checkpoint_id.height)
            .expect("the tracker did not have a checkpoint at that height");
        assert_eq!(
            data.block_hash, checkpoint_id.hash,
            "tracker had a different block hash for checkpoint at that height"
        );

        sha256::Hash::from_engine(data.accum_digest.clone())
    }

    /// Get the [BlockId] for the last known tip.
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

    /// Get the earliest checkpoint whose height is greater than or equal to height if it exists.
    fn checkpoint_covering(&self, height: u32) -> Option<BlockId> {
        let (cp_height, data) = self.checkpoints.range(height..).next()?;
        Some(BlockId {
            height: *cp_height,
            hash: data.block_hash,
        })
    }

    /// Return an iterator over [BlockId] from newest to oldest, for this tracker
    pub fn iter_checkpoints(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.checkpoints.iter().rev().map(|(height, data)| BlockId {
            height: *height,
            hash: data.block_hash,
        })
    }

    fn remove_tx(&mut self, txid: Txid) {
        let aug_tx = match self.txs.remove(&txid) {
            Some(aug_tx) => aug_tx,
            None => {
                debug_assert!(!self.mempool.contains(&txid), "Consistency check");
                return;
            }
        };

        // Input processing
        for input in &aug_tx.tx.input {
            if let Some((_, tx_that_spends)) = self.spends.remove(&input.previous_output) {
                debug_assert_eq!(
                    tx_that_spends, txid,
                    "the one that spent it must be this one"
                );
            }

            if self.txouts.contains_key(&input.previous_output) {
                self.unspent.insert(input.previous_output);
            }
        }

        // Output Processing
        for i in 0..aug_tx.tx.output.len() {
            let txout_to_remove = OutPoint {
                vout: i as u32,
                txid,
            };
            if let Some(derivation_index) = self.txouts.remove(&txout_to_remove) {
                self.script_txouts
                    .get_mut(&derivation_index)
                    .expect("guaranteed to exist")
                    .remove(&txout_to_remove);

                // TODO: Decide if we should enforce reversal of "used" script_pubkeys into "unused".
            }
        }

        self.mempool.remove(&txid);
    }

    // Returns the checkpoint height at which it got added so we can recompute the txid digest from
    // that point.
    fn add_tx(
        &mut self,
        inputs: PrevOuts,
        tx: Transaction,
        confirmation_time: Option<BlockTime>,
    ) -> Option<u32> {
        let txid = tx.txid();

        let inputs_sum = match inputs {
            PrevOuts::Coinbase => {
                debug_assert_eq!(tx.input.len(), 1);
                debug_assert!(tx.input[0].previous_output.is_null());
                // HACK: set to 0. We only use this for fee which for coinbase is always 0.
                0
            }
            PrevOuts::Spend(txouts) => txouts.iter().map(|output| output.value).sum(),
        };

        let outputs_sum: u64 = tx.output.iter().map(|out| out.value).sum();
        // we need to saturating sub since we want coinbase txs to map to 0 fee and
        // this subtraction will be negative for coinbase txs.
        let fee = inputs_sum.saturating_sub(outputs_sum);
        let feerate = fee as f32 / tx.weight() as f32;

        // Look for conflicts to determine whether we should add this transaction or remove the one
        // it conflicts with. Note that the txids in conflicts will always be unconfirmed
        // transactions (dealing with confirmed conflicts is done outside and is usually an error).
        let conflicts = tx
            .input
            .iter()
            .filter_map(|input| {
                self.spends
                    .get(&input.previous_output)
                    .map(|(_, txid)| *txid)
            })
            .collect::<Vec<_>>();

        if confirmation_time.is_some() {
            // Because we made sure we only have mempool transactions in conflict list, if this one
            // is already confirmed, its safe to remove them.
            for conflicting_txid in conflicts {
                self.remove_tx(conflicting_txid);
            }
        } else {
            let conflicing_tx_with_higher_feerate = conflicts.iter().find(|conflicting_txid| {
                self.txs.get(*conflicting_txid).expect("must exist").feerate > feerate
            });
            if conflicing_tx_with_higher_feerate.is_none() {
                for conflicting_txid in conflicts {
                    self.remove_tx(conflicting_txid);
                }
            } else {
                // we shouldn't add this tx as it conflicts with one with a higher feerate.
                return None;
            }
        }

        for (i, input) in tx.input.iter().enumerate() {
            let removed = self.spends.insert(input.previous_output, (i as u32, txid));
            debug_assert_eq!(
                removed, None,
                "we should have already removed all conflicts!"
            );
            self.unspent.remove(&input.previous_output);
        }

        for (i, out) in tx.output.iter().enumerate() {
            if let Some(index) = self.index_of_derived_script(&out.script_pubkey) {
                let outpoint = OutPoint {
                    txid,
                    vout: i as u32,
                };

                self.txouts.insert(outpoint, index);

                if !self.spends.contains_key(&outpoint) {
                    self.unspent.insert(outpoint);
                }

                let txos_for_script = self.script_txouts.entry(index).or_default();
                txos_for_script.insert(outpoint);
                self.unused.remove(&index);
            }
        }

        // If all goes well, add this into out txs list
        self.txs.insert(
            txid,
            AugmentedTx {
                tx,
                fee,
                feerate,
                confirmation_time,
            },
        );

        // If this Tx is confirmed add it into the right CheckpointData
        // If not, add it into mempool.
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
        let txs_to_remove = removed
            .values()
            .rev()
            .map(|data| data.ordered_txids.iter().map(|(_, txid)| txid))
            .flatten();
        for tx_to_remove in txs_to_remove {
            self.remove_tx(*tx_to_remove);
        }
    }

    /// Applies a new candidate checkpoint to the tracker.
    pub fn apply_checkpoint(&mut self, mut new_checkpoint: CheckpointCandidate) -> ApplyResult {
        // Do consistency checks first so we don't mutate anything until we're sure the update is
        // valid. We check for two things
        // 1. There's no "known" transaction in the new checkpoint with same txid but different conf_time
        // 2. No transaction double spends one of our existing confirmed transactions.

        // We simply ignore transactions in the checkpoint that have a confirmation time greater
        // than the checkpoint height. I felt this was better for the caller than creating an error
        // type.
        new_checkpoint
            .transactions
            .retain(|(_, _, confirmation_time)| {
                if let Some(confirmation_time) = confirmation_time {
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

        for (_, tx, confirmation_time) in &new_checkpoint.transactions {
            let txid = tx.txid();
            if let Some(existing) = self.txs.get(&tx.txid()) {
                if let Some(existing_time) = existing.confirmation_time {
                    // no need to consider conflicts for txs that are about to be invalidated
                    if existing_time.height >= invalidation_height {
                        continue;
                    }
                    if confirmation_time != &Some(existing_time) {
                        if existing_time.height < invalidation_height {
                            let at_checkpoint = self
                                .checkpoint_covering(existing_time.height)
                                .expect("must exist since there's a confirmed tx");

                            return ApplyResult::Inconsistent {
                                txid,
                                conflicts_with: existing.tx.txid(),
                                at_checkpoint,
                            };
                        }
                    }
                }
            }

            let conflicts = tx
                .input
                .iter()
                .filter_map(|input| self.spends.get(&input.previous_output));
            for (_, conflicting_txid) in conflicts {
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

                    let at_checkpoint = self
                        .checkpoint_covering(conflicting_conftime.height)
                        .expect("must exist since there's a confirmed tx");

                    return ApplyResult::Inconsistent {
                        txid,
                        conflicts_with: *conflicting_txid,
                        at_checkpoint,
                    };
                }
            }
        }

        match new_checkpoint.invalidate {
            Some(checkpoint_reset) => match self.checkpoints.get(&checkpoint_reset.height) {
                Some(existing_checkpoint) => {
                    if existing_checkpoint.block_hash != checkpoint_reset.hash {
                        if self
                            .checkpoints
                            .range(..checkpoint_reset.height)
                            .last()
                            .map(|(height, data)| BlockId {
                                height: *height,
                                hash: data.block_hash,
                            })
                            == new_checkpoint.base_tip
                        {
                            self.invalidate_checkpoint(checkpoint_reset.height);
                        } else {
                            return ApplyResult::Stale;
                        }
                    } else {
                        return ApplyResult::Stale;
                    }
                }
                None => return ApplyResult::Stale,
            },
            None => {
                if new_checkpoint.base_tip != self.latest_checkpoint() {
                    return ApplyResult::Stale;
                }
            }
        }

        self.checkpoints
            .entry(new_checkpoint.new_tip.height)
            .or_insert_with(|| CheckpointData {
                block_hash: new_checkpoint.new_tip.hash,
                ..Default::default()
            });

        let mut deepest_change = None;
        for (vouts, tx, confirmation_time) in new_checkpoint.transactions {
            if let Some(change) = self.add_tx(vouts, tx, confirmation_time) {
                deepest_change = Some(deepest_change.unwrap_or(u32::MAX).min(change));
            }
        }

        if let Some(change_depth) = deepest_change {
            self.recompute_txid_digests(change_depth);
        }

        // If no new transactions were added in new_tip, remove it.
        if self
            .checkpoints
            .values()
            .rev()
            .next()
            .unwrap()
            .ordered_txids
            .is_empty()
        {
            self.checkpoints.remove(&new_checkpoint.new_tip.height);
        }

        self.last_tip_seen = Some(new_checkpoint.new_tip);

        self.apply_checkpoint_limit();
        debug_assert!(self.is_latest_checkpoint_hash_correct());

        ApplyResult::Ok
    }

    /// Performs recomputation of transaction digest of checkpoint data
    /// from the given height.
    fn recompute_txid_digests(&mut self, from: u32) {
        let mut prev_accum_digest = self
            .checkpoints
            .range(..from)
            .last()
            .map(|(_, prev)| prev.accum_digest.clone())
            .unwrap_or_else(sha256::HashEngine::default);

        for (_height, data) in self.checkpoints.range_mut(from..) {
            let mut accum_digest = prev_accum_digest.clone();
            for (_, txid) in &data.ordered_txids {
                accum_digest.input(txid);
            }
            data.accum_digest = accum_digest.clone();
            prev_accum_digest = accum_digest;
        }
    }

    /// Takes the checkpoint at a height and merges its transactions into the next checkpoint
    pub fn merge_checkpoint(&mut self, height: u32) {
        if let Some(checkpoint) = self.checkpoints.remove(&height) {
            match self.checkpoints.range_mut((height + 1)..).next() {
                Some((_, next_one)) => next_one.ordered_txids.extend(checkpoint.ordered_txids),
                None => {
                    // put it back because there's only one checkpoint.
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
    pub fn disconnect_block(&mut self, block_height: u32, block_hash: BlockHash) {
        // Can't guarantee that mempool is consistent with chain after we disconnect a block so we
        // clear it.
        // TODO: it would be nice if we could only delete those transactions that are
        // inconsistent by recording the latest block they were included in.
        self.clear_mempool();
        if let Some(checkpoint_data) = self.checkpoints.get(&block_height) {
            if checkpoint_data.block_hash == block_hash {
                self.invalidate_checkpoint(block_height);
            }
        }
    }

    /// Iterate over all transactions in our transaction store.
    /// Can be both related/unrelated and/or confirmed/unconfirmed.
    pub fn iter_tx(&self) -> impl Iterator<Item = (Txid, &AugmentedTx)> {
        self.txs.iter().map(|(txid, tx)| (*txid, tx))
    }

    /// Iterates over all transactions related to the descriptor ordered by decending confirmation
    /// with those transactions that are unconfirmed first.
    ///
    /// "related" means that the transactoin has an output with a script pubkey produced by the
    /// descriptor or it spends from such an output.
    pub fn iter_tx_by_confirmation_time(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Txid, &AugmentedTx)> + '_ {
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

    /// Iterate over unspent [LocalTxOut]s
    pub fn iter_unspent(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.unspent
            .iter()
            .map(|txo| (txo, self.txouts.get(txo).expect("txout must exist")))
            .map(|(txo, index)| self.create_txout(*txo, *index))
    }

    // Create a [LocalTxOut] given an outpoint and derivation index
    fn create_txout(&self, outpoint: OutPoint, derivation_index: u32) -> LocalTxOut {
        let tx = self
            .txs
            .get(&outpoint.txid)
            .expect("must exist since we have the txout");
        let spent_by = self.spends.get(&outpoint).cloned();
        let value = self.txs.get(&outpoint.txid).expect("must exist").tx.output
            [outpoint.vout as usize]
            .value;
        LocalTxOut {
            value,
            spent_by,
            outpoint,
            derivation_index,
            confirmed_at: tx.confirmation_time,
        }
    }

    /// Iterate over all the transaction outputs discovered by the tracker with script pubkeys
    /// matches those stored by the tracker.
    pub fn iter_txout(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.txouts
            .iter()
            .map(|(outpoint, data)| self.create_txout(*outpoint, *data))
    }

    /// Get a transaction output at given outpoint.
    pub fn get_txout(&self, outpoint: OutPoint) -> Option<LocalTxOut> {
        let data = self.txouts.get(&outpoint)?;
        Some(self.create_txout(outpoint, *data))
    }

    /// Get a stored transaction given its `Txid`.
    pub fn get_tx(&self, txid: Txid) -> Option<&AugmentedTx> {
        self.txs.get(&txid)
    }

    /// Returns the script that has been derived at the index.
    ///
    /// If that index hasn't been derived yet it will return `None`.
    pub fn script_at_index(&self, index: u32) -> Option<&Script> {
        self.scripts.get(index as usize)
    }

    /// Iterate over the scripts that have been derived already
    pub fn scripts(&self) -> &[Script] {
        &self.scripts
    }

    /// Adds a script to the script tracker.
    ///
    /// The tracker will look for transactions spending to/from this scriptpubkey on all checkpoints
    /// that are subsequently added.
    pub fn add_script(&mut self, script: Script) -> usize {
        let index = self.scripts.len();
        self.script_indexes.insert(script.clone(), index as u32);
        self.scripts.push(script);
        self.unused.insert(index as u32);
        index
    }

    /// Iterate over the scripts that have been derived but do not have a transaction spending to them.
    pub fn iter_unused_scripts(&self) -> impl Iterator<Item = (u32, &Script)> {
        self.unused
            .iter()
            .map(|index| (*index, self.script_at_index(*index).expect("must exist")))
    }

    /// Returns whether the script at index `index` has been used or not.
    ///
    /// Will also return `false` if the script at `index` hasn't been derived yet (because we have
    /// no way of knowing if it has been used yet in that case).
    pub fn is_used(&self, index: u32) -> bool {
        !self.unused.contains(&index) && (index as usize) < self.scripts.len()
    }

    /// Returns at what derivation index a script pubkey was derived at.
    pub fn index_of_derived_script(&self, script: &Script) -> Option<u32> {
        self.script_indexes.get(script).cloned()
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
            self.accum_digest_at(tip) == txid_hash
        } else {
            true
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCandidate {
    /// List of transactions in this checkpoint. They needs to be consistent with tracker's state
    /// for the new checkpoint to be included.
    pub transactions: Vec<(PrevOuts, Transaction, Option<BlockTime>)>,
    /// Update the last active index of the tracker to given value.
    pub last_active_index: Option<u32>,
    /// The new checkpoint can be applied upon this tip. A tracker will usually reject updates that
    /// do not have `base_tip` equal to it's latest valid checkpoint.
    pub base_tip: Option<BlockId>,
    /// Invalidates a checkpoint before considering this checkpoint.
    pub invalidate: Option<BlockId>,
    /// Sets the tip that this checkpoint was creaed for. All data in this checkpoint must be valid
    /// with respect to this tip.
    pub new_tip: BlockId,
}

/// A transaction with extra metadata
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedTx {
    pub tx: Transaction,
    pub fee: u64,
    pub feerate: f32,
    pub confirmation_time: Option<BlockTime>,
}

/// An UTXO with extra metadata
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTxOut {
    pub value: u64,
    pub spent_by: Option<(u32, Txid)>,
    pub outpoint: OutPoint,
    pub derivation_index: u32,
    pub confirmed_at: Option<BlockTime>,
}

#[derive(Debug, Clone)]
pub struct PrimedInput {
    pub descriptor: Descriptor<DefiniteDescriptorKey>,
    pub psbt_input: psbt::Input,
}
