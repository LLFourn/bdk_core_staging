use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use crate::{BlockId, BlockTime, Vec};
use bitcoin::secp256k1::Secp256k1;
use bitcoin::util::address::WitnessVersion;
use bitcoin::{
    self,
    hashes::{sha256, Hash, HashEngine},
    util::psbt::{self, PartiallySignedTransaction as Psbt},
    BlockHash, OutPoint, Script, Transaction, Txid,
};
use bitcoin::{Block, TxIn, TxOut};
use miniscript::psbt::PsbtInputExt;
use miniscript::{DefiniteDescriptorKey, Descriptor, DescriptorPublicKey};

#[derive(Clone, Debug)]
pub struct ScriptTracker<I> {
    /// Checkpoint data (txids)
    checkpoints: BTreeMap<u32, CheckpointData>,
    /// Index the Outpoints owned by this tracker to the derivation index of script pubkey.
    txouts: BTreeMap<OutPoint, I>,
    /// List of all known spends. Including our's and other's outpoints. Both confirmed and unconfirmed.
    /// This is useful to track all inputs we might ever care about.
    spends: BTreeMap<OutPoint, (u32, Txid)>,
    /// Set of our unspent outpoints.
    unspent: HashSet<OutPoint>,
    /// Derived script_pubkeys ordered by derivation index.
    scripts: BTreeMap<I, Script>,
    /// A reverse lookup from out script_pubkeys to derivation index
    script_indexes: HashMap<Script, I>,
    /// A lookup from script pubkey derivation index to related outpoints
    script_txouts: BTreeMap<I, HashSet<OutPoint>>,
    /// A set of unused derivation indices.
    unused: BTreeSet<I>,
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

impl<I> Default for ScriptTracker<I> {
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

impl<I: Clone + Ord> ScriptTracker<I> {
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
    fn add_tx(&mut self, tx: Transaction, confirmation_time: Option<BlockTime>) -> Option<u32> {
        let txid = tx.txid();

        // let inputs_sum = match inputs {
        //     PrevOuts::Coinbase => {
        //         debug_assert_eq!(tx.input.len(), 1);
        //         debug_assert!(tx.input[0].previous_output.is_null());
        //         // HACK: set to 0. We only use this for fee which for coinbase is always 0.
        //         0
        //     }
        //     PrevOuts::Spend(txouts) => txouts.iter().map(|output| output.value).sum(),
        // };

        // let outputs_sum: u64 = tx.output.iter().map(|out| out.value).sum();
        // // we need to saturating sub since we want coinbase txs to map to 0 fee and
        // // this subtraction will be negative for coinbase txs.
        // let fee = inputs_sum.saturating_sub(outputs_sum);
        // let feerate = fee as f32 / tx.weight() as f32;

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
            let conflicing_tx_with_higher_feerate = conflicts.iter().find(|_conflicting_txid| {
                // self.txs.get(*conflicting_txid).expect("must exist").feerate > feerate
                false
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
            if let Some(index) = self.index_of_script(&out.script_pubkey).cloned() {
                let outpoint = OutPoint {
                    txid,
                    vout: i as u32,
                };

                self.txouts.insert(outpoint, index.clone());

                if !self.spends.contains_key(&outpoint) {
                    self.unspent.insert(outpoint);
                }

                let txos_for_script = self.script_txouts.entry(index.clone()).or_default();
                txos_for_script.insert(outpoint);
                self.unused.remove(&index);
            }
        }

        // If all goes well, add this into out txs list
        self.txs.insert(
            txid,
            AugmentedTx {
                tx,
                // fee,
                // feerate,
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

    pub fn apply_block(&mut self, height: u32, block: Block) -> ApplyResult {
        let hash = block.block_hash();
        let prev_hash = block.header.prev_blockhash;

        if let Some(prev_tip) = self.last_tip_seen() {
            if prev_tip.hash != prev_hash {
                return ApplyResult::Stale;
            }
        }

        let txs = block
            .txdata
            .into_iter()
            .filter(|tx| {
                tx.input
                    .iter()
                    .any(|txin| self.txouts.get(&txin.previous_output).is_some())
                    || tx
                        .output
                        .iter()
                        .any(|txout| self.script_indexes.contains_key(&txout.script_pubkey))
            })
            .map(|tx| {
                (
                    tx,
                    Some(BlockTime {
                        height,
                        time: block.header.time,
                    }),
                )
            })
            .collect();

        let candidate = CheckpointCandidate {
            transactions: txs,
            base_tip: self.last_tip_seen,
            invalidate: None,
            new_tip: BlockId { height, hash },
        };

        self.apply_checkpoint(candidate)
    }

    /// Applies a new candidate checkpoint to the tracker.
    #[must_use]
    pub fn apply_checkpoint(&mut self, mut new_checkpoint: CheckpointCandidate) -> ApplyResult {
        new_checkpoint
            .transactions
            .retain(|(_, confirmation_time)| {
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

        // Do consistency checks first so we don't mutate anything until we're sure the update is
        // valid. We check for two things
        // 1. There's no "known" transaction in the new checkpoint with same txid but different conf_time
        // 2. No transaction double spends one of our existing confirmed transactions.

        // We simply ignore transactions in the checkpoint that have a confirmation time greater
        // than the checkpoint height. I felt this was better for the caller than creating an error
        // type.

        for (tx, confirmation_time) in &new_checkpoint.transactions {
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
                .filter_map(|input| self.spends.get(&input.previous_output))
                .filter(|(_, conflict_txid)| *conflict_txid != tx.txid());

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
        for (tx, confirmation_time) in new_checkpoint.transactions {
            if let Some(change) = self.add_tx(tx, confirmation_time) {
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
    pub fn iter_unspent(&self) -> impl Iterator<Item = LocalTxOut<I>> + '_ {
        self.unspent
            .iter()
            .map(|txo| (txo, self.txouts.get(txo).expect("txout must exist")))
            .map(|(txo, index)| self.create_txout(*txo, index.clone()))
    }

    // Create a [LocalTxOut] given an outpoint and derivation index
    fn create_txout(&self, outpoint: OutPoint, spk_index: I) -> LocalTxOut<I> {
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
            spk_index,
            confirmed_at: tx.confirmation_time,
        }
    }

    /// Iterate over all the transaction outputs discovered by the tracker with script pubkeys
    /// matches those stored by the tracker.
    pub fn iter_txout(&self) -> impl Iterator<Item = LocalTxOut<I>> + '_ {
        self.txouts
            .iter()
            .map(|(outpoint, script_index)| self.create_txout(*outpoint, script_index.clone()))
    }

    /// Get a transaction output at given outpoint.
    pub fn get_txout(&self, outpoint: OutPoint) -> Option<LocalTxOut<I>> {
        let data = self.txouts.get(&outpoint)?;
        Some(self.create_txout(outpoint, data.clone()))
    }

    /// Get a stored transaction given its `Txid`.
    pub fn get_tx(&self, txid: Txid) -> Option<&AugmentedTx> {
        self.txs.get(&txid)
    }

    /// Returns the script that has been derived at the index.
    ///
    /// If that index hasn't been derived yet it will return `None`.
    pub fn script_at_index(&self, index: &I) -> Option<&Script> {
        self.scripts.get(index)
    }

    /// Iterate over the scripts that have been derived already
    pub fn scripts(&self) -> &BTreeMap<I, Script> {
        &self.scripts
    }

    /// Adds a script to the script tracker.
    ///
    /// The tracker will look for transactions spending to/from this scriptpubkey on all checkpoints
    /// that are subsequently added.
    pub fn add_script(&mut self, index: I, spk: Script) {
        self.script_indexes.insert(spk.clone(), index.clone());
        self.scripts.insert(index.clone(), spk);
        self.unused.insert(index);
    }

    /// Iterate over the scripts that have been derived but do not have a transaction spending to them.
    pub fn iter_unused_scripts(&self) -> impl Iterator<Item = (I, &Script)> {
        self.unused.iter().map(|index| {
            (
                index.clone(),
                self.script_at_index(index).expect("must exist"),
            )
        })
    }

    /// Returns whether the script at index `index` has been used or not.
    ///
    /// i.e. has a transaction which spends to it.
    pub fn is_used(&self, index: I) -> bool {
        self.script_txouts
            .get(&index)
            .map(|set| !set.is_empty())
            .unwrap_or(false)
    }

    /// Returns at what derivation index a script pubkey was derived at.
    pub fn index_of_script(&self, script: &Script) -> Option<&I> {
        self.script_indexes.get(script)
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

    pub fn create_psbt(
        &self,
        inputs: impl IntoIterator<Item = OutPoint>,
        outputs: impl IntoIterator<Item = TxOut>,
        lookup_descriptor: impl Fn(I) -> Option<Descriptor<DefiniteDescriptorKey>>,
    ) -> (Psbt, BTreeMap<usize, Descriptor<DefiniteDescriptorKey>>) {
        let unsigned_tx = Transaction {
            version: 0x01,
            lock_time: 0x00,
            input: inputs
                .into_iter()
                .map(|previous_output| TxIn {
                    previous_output,
                    ..Default::default()
                })
                .collect(),
            output: outputs.into_iter().collect(),
        };

        let mut psbt = Psbt::from_unsigned_tx(unsigned_tx).unwrap();
        let mut definite_descriptors = BTreeMap::new();

        for ((input_index, psbt_input), txin) in psbt
            .inputs
            .iter_mut()
            .enumerate()
            .zip(&psbt.unsigned_tx.input)
        {
            let definite_descriptor = self
                .txouts
                .get(&txin.previous_output)
                .cloned()
                .and_then(|index| lookup_descriptor(index));
            if let Some(definite_descriptor) = definite_descriptor {
                let prev_tx = self
                    .txs
                    .get(&txin.previous_output.txid)
                    .expect("since the txout exists so must the transaction");
                match definite_descriptor.desc_type().segwit_version() {
                    Some(version) => {
                        if version < WitnessVersion::V1 {
                            psbt_input.non_witness_utxo = Some(prev_tx.tx.clone());
                        }
                        psbt_input.witness_utxo =
                            Some(prev_tx.tx.output[txin.previous_output.vout as usize].clone());
                    }
                    None => psbt_input.non_witness_utxo = Some(prev_tx.tx.clone()),
                }

                psbt_input
                    .update_with_descriptor_unchecked(&definite_descriptor)
                    .expect("conversion error cannot happen if descriptor is well formed");
                definite_descriptors.insert(input_index, definite_descriptor);
            }
        }

        (psbt, definite_descriptors)
    }
}

#[derive(Debug, Clone)]
pub struct PrimedInput {
    pub descriptor: Descriptor<DefiniteDescriptorKey>,
    pub psbt_input: psbt::Input,
}

impl ScriptTracker<u32> {
    /// Derives scriptpubkeys from the descriptor **up to and including** `end` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn derive_scripts(
        &mut self,
        descriptor: &Descriptor<DescriptorPublicKey>,
        end: u32,
    ) -> bool {
        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => end,
        };

        let needed = (end + 1).saturating_sub(self.scripts.len() as u32);
        for index in self.scripts.len()..self.scripts.len() + needed as usize {
            let spk = descriptor
                .at_derivation_index(index as u32)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            self.add_script(index as u32, spk);
        }

        needed == 0
    }

    /// Derives a new script pubkey which can be turned into an address.
    ///
    /// The tracker returns a new address for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn derive_new(&mut self, descriptor: &Descriptor<DescriptorPublicKey>) -> (u32, &Script) {
        let next_derivation_index = if descriptor.has_wildcard() {
            self.scripts.len() as u32
        } else {
            0
        };
        self.derive_scripts(descriptor, next_derivation_index);
        let script = self
            .scripts
            .get(&next_derivation_index)
            .expect("we just derived to that index");
        (next_derivation_index, script)
    }

    /// Derives and stores a new scriptpubkey only if we haven't already got one that hasn't received any
    /// coins yet.
    pub fn derive_next_unused(
        &mut self,
        descriptor: &Descriptor<DescriptorPublicKey>,
    ) -> (u32, &Script) {
        let need_new = self.iter_unused_scripts().next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(descriptor)
        } else {
            self.iter_unused_scripts()
                .map(|(i, script)| (i, script))
                .next()
                .unwrap()
        }
    }
}

impl<K: Ord + Clone> ScriptTracker<(K, u32)> {
    /// Derives scriptpubkeys from the descriptor **up to and including** `end` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn keychain_derive_scripts(
        &mut self,
        keychain: K,
        descriptor: &Descriptor<DescriptorPublicKey>,
        end: u32,
    ) -> bool {
        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => end,
        };
        let next_to_derive = self
            .scripts
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .last()
            .map(|((_, last), _)| last + 1)
            .unwrap_or(0);

        if next_to_derive > end {
            return false;
        }

        for index in next_to_derive..=end {
            let spk = descriptor
                .at_derivation_index(index as u32)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            self.add_script((keychain.clone(), index as u32), spk);
        }

        true
    }

    /// Derives a new scriptpubkey for a keychain.
    ///
    /// The tracker returns a new scriptpubkey for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn keychain_derive_new(
        &mut self,
        keychain: K,
        descriptor: &Descriptor<DescriptorPublicKey>,
    ) -> (u32, &Script) {
        let next_derivation_index = if descriptor.has_wildcard() {
            self.scripts.len() as u32
        } else {
            0
        };
        self.keychain_derive_scripts(keychain.clone(), descriptor, next_derivation_index);
        let script = self
            .scripts
            .get(&(keychain, next_derivation_index))
            .expect("we just derived to that index");
        (next_derivation_index, script)
    }

    pub fn keychain_derive_next_unused(
        &mut self,
        keychain: K,
        descriptor: &Descriptor<DescriptorPublicKey>,
    ) -> (u32, &Script) {
        let need_new = self.iter_unused_scripts().next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.keychain_derive_new(keychain, descriptor)
        } else {
            self.iter_unused_scripts()
                .filter(|((kc, _), _)| kc == &keychain)
                .map(|((_, i), script)| (i, script))
                .next()
                .unwrap()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointCandidate {
    /// List of transactions in this checkpoint. They needs to be consistent with tracker's state
    /// for the new checkpoint to be included.
    pub transactions: Vec<(Transaction, Option<BlockTime>)>,
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
    pub confirmation_time: Option<BlockTime>,
}

/// An UTXO with extra metadata
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTxOut<I> {
    pub value: u64,
    pub spent_by: Option<(u32, Txid)>,
    pub outpoint: OutPoint,
    pub spk_index: I,
    pub confirmed_at: Option<BlockTime>,
}

#[cfg(test)]
mod test {
    use bitcoin::{secp256k1::Secp256k1, BlockHash, Transaction, TxIn, TxOut};
    use miniscript::{Descriptor, DescriptorPublicKey};

    use super::*;

    const DESCRIPTOR: &'static str = "wpkh(xpub6ERApfZwUNrhLCkDtcHTcxd75RbzS1ed54G1LkBUHQVHQKqhMkhgbmJbZRkrgZw4koxb5JaHWkY4ALHY2grBGRjaDMzQLcgJvLJuZZvRcEL)";

    pub enum IOSpec {
        Mine(/* value */ u64, /* the derivation index */ usize),
        Other(/*value*/ u64),
    }

    pub struct TxSpec {
        inputs: Vec<IOSpec>,
        outputs: Vec<IOSpec>,
        confirmed_at: Option<u32>,
        is_coinbase: bool,
    }

    #[derive(Clone, Debug)]
    struct UpdateGen {
        pub vout_counter: u32,
        pub prev_tip: Option<BlockId>,
        pub descriptor: Descriptor<DescriptorPublicKey>,
    }

    impl UpdateGen {
        pub fn new() -> Self {
            Self {
                vout_counter: 0,
                prev_tip: None,
                descriptor: DESCRIPTOR.parse().unwrap(),
            }
        }

        fn next_txin(&mut self) -> TxIn {
            let txin = TxIn {
                previous_output: OutPoint {
                    txid: Txid::default(),
                    vout: self.vout_counter,
                },
                ..Default::default()
            };
            self.vout_counter += 1;
            txin
        }

        fn create_update(
            &mut self,
            txs: Vec<TxSpec>,
            checkpoint_height: u32,
        ) -> CheckpointCandidate {
            let secp = Secp256k1::verification_only();

            let transactions = txs
                .into_iter()
                .map(|tx_spec| {
                    (
                        // match tx_spec.is_coinbase {
                        //     false => PrevOuts::Spend(
                        //         tx_spec
                        //             .inputs
                        //             .iter()
                        //             .map(|in_spec| match in_spec {
                        //                 IOSpec::Mine(value, index) => TxOut {
                        //                     value: *value,
                        //                     script_pubkey: self
                        //                         .descriptor
                        //                         .at_derivation_index(*index as u32)
                        //                         .derived_descriptor(&secp)
                        //                         .unwrap()
                        //                         .script_pubkey(),
                        //                 },
                        //                 IOSpec::Other(value) => TxOut {
                        //                     value: *value,
                        //                     script_pubkey: Default::default(),
                        //                 },
                        //             })
                        //             .collect(),
                        //     ),
                        //     true => {
                        //         todo!()
                        //     }
                        // },
                        Transaction {
                            version: 1,
                            lock_time: 0,
                            input: if tx_spec.is_coinbase {
                                todo!()
                            } else {
                                tx_spec.inputs.iter().map(|_| self.next_txin()).collect()
                            },
                            output: tx_spec
                                .outputs
                                .into_iter()
                                .map(|out_spec| match out_spec {
                                    IOSpec::Other(value) => TxOut {
                                        value,
                                        script_pubkey: Script::default(),
                                    },
                                    IOSpec::Mine(value, index) => TxOut {
                                        value,
                                        script_pubkey: self
                                            .descriptor
                                            .at_derivation_index(index as u32)
                                            .derived_descriptor(&secp)
                                            .unwrap()
                                            .script_pubkey(),
                                    },
                                })
                                .collect(),
                        },
                        tx_spec.confirmed_at.map(|confirmed_at| BlockTime {
                            height: confirmed_at,
                            time: confirmed_at,
                        }),
                    )
                })
                .collect();

            let new_tip = BlockId {
                height: checkpoint_height,
                hash: BlockHash::default(),
            };

            let update = CheckpointCandidate {
                transactions,
                new_tip,
                invalidate: None,
                base_tip: self.prev_tip,
            };

            self.prev_tip = Some(new_tip);

            update
        }
    }

    #[test]
    fn no_checkpoint_and_then_confirm() {
        use IOSpec::*;

        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::default();
        tracker.derive_scripts(&update_gen.descriptor, 0);

        let mut checkpoint = update_gen.create_update(
            vec![TxSpec {
                inputs: vec![Other(2_000)],
                outputs: vec![Mine(1_000, 0), Other(1_800)],
                confirmed_at: None,
                is_coinbase: false,
            }],
            0,
        );

        assert_eq!(
            tracker.apply_checkpoint(checkpoint.clone()),
            ApplyResult::Ok
        );

        let txouts = tracker.iter_txout().collect::<Vec<_>>();
        let txs = tracker.iter_tx().collect::<Vec<_>>();
        let unspent = tracker.iter_unspent().collect::<Vec<_>>();
        let checkpoints = tracker.iter_checkpoints().collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        assert_eq!(unspent, txouts);
        assert_eq!(txs.len(), 1);
        assert_eq!(checkpoints.len(), 0);
        assert_eq!(txouts.len(), 1);

        checkpoint.transactions[0].1 = Some(BlockTime { height: 1, time: 1 });
        checkpoint.new_tip = BlockId {
            height: checkpoint.new_tip.height + 1,
            hash: checkpoint.new_tip.hash,
        };

        assert_eq!(tracker.apply_checkpoint(checkpoint), ApplyResult::Ok);

        let txs = tracker.iter_tx().collect::<Vec<_>>();
        let checkpoints = tracker.iter_checkpoints().collect::<Vec<_>>();
        let txouts = tracker.iter_txout().collect::<Vec<_>>();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(txouts.len(), 1);
        assert_eq!(
            tracker.checkpoint_txids(checkpoints[0]).collect::<Vec<_>>(),
            txs.iter().map(|(x, _)| *x).collect::<Vec<_>>()
        );
        assert!(tracker.is_latest_checkpoint_hash_correct());
    }

    #[test]
    fn two_checkpoints_then_merege() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();
        tracker.derive_scripts(&update_gen.descriptor, 3);

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                vec![
                    TxSpec {
                        inputs: vec![Other(2_000)],
                        outputs: vec![Mine(2_000, 0)],
                        confirmed_at: Some(1),
                        is_coinbase: false,
                    },
                    TxSpec {
                        inputs: vec![Other(1_000)],
                        outputs: vec![Mine(1_000, 1)],
                        confirmed_at: Some(0),
                        is_coinbase: false,
                    },
                ],
                1,
            )),
            ApplyResult::Ok
        );

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                vec![
                    TxSpec {
                        inputs: vec![Other(3_000)],
                        outputs: vec![Mine(3_000, 2)],
                        confirmed_at: Some(2),
                        is_coinbase: false,
                    },
                    TxSpec {
                        inputs: vec![Other(4_000)],
                        outputs: vec![Mine(4_000, 3)],
                        confirmed_at: Some(3),
                        is_coinbase: false,
                    },
                ],
                3,
            )),
            ApplyResult::Ok
        );

        assert_eq!(tracker.iter_txout().count(), 4);

        // there is no checkpoint here
        tracker.merge_checkpoint(0);
        assert_eq!(tracker.iter_checkpoints().count(), 2);

        tracker.merge_checkpoint(1);
        assert_eq!(tracker.iter_checkpoints().count(), 1);

        let txids = tracker.checkpoint_txids(tracker.checkpoint_at(3).unwrap());
        assert_eq!(txids.count(), 4);
    }

    #[test]
    fn invalid_tx_confirmation_time() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                vec![TxSpec {
                    inputs: vec![Other(2_000)],
                    outputs: vec![Mine(2_000, 1)],
                    confirmed_at: Some(2),
                    is_coinbase: false,
                },],
                1,
            )),
            ApplyResult::Ok
        );

        assert_eq!(tracker.iter_checkpoints().count(), 0);
        assert_eq!(tracker.iter_tx().count(), 0);
    }

    #[test]
    fn out_of_order_tx_is_before_first_checkpoint() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                vec![TxSpec {
                    inputs: vec![Other(2_000)],
                    outputs: vec![Mine(2_000, 1)],
                    confirmed_at: Some(1),
                    is_coinbase: false,
                },],
                1,
            )),
            ApplyResult::Ok
        );

        assert!(tracker.is_latest_checkpoint_hash_correct());

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                vec![TxSpec {
                    inputs: vec![Other(2_000)],
                    outputs: vec![Mine(2_000, 1)],
                    confirmed_at: Some(0),
                    is_coinbase: false,
                },],
                2,
            )),
            ApplyResult::Ok
        );
    }

    #[test]
    fn checkpoint_limit_is_applied() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();
        tracker.set_checkpoint_limit(5);

        for i in 0..10 {
            assert_eq!(
                tracker.apply_checkpoint(update_gen.create_update(
                    vec![TxSpec {
                        inputs: vec![Other(2_000)],
                        outputs: vec![Mine(2_000, i)],
                        confirmed_at: Some(i as u32),
                        is_coinbase: false,
                    },],
                    i as u32,
                )),
                ApplyResult::Ok
            );
        }

        assert_eq!(tracker.iter_tx().count(), 10);
        assert_eq!(tracker.iter_checkpoints().count(), 5);
    }

    #[test]
    fn many_transactions_in_the_same_height() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();
        let txs = (0..100)
            .map(|_| TxSpec {
                inputs: vec![Other(1_900)],
                outputs: vec![Mine(2_000, 0)],
                confirmed_at: Some(1),
                is_coinbase: false,
            })
            .collect();

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(txs, 1,)),
            ApplyResult::Ok
        );
    }

    #[test]
    fn same_checkpoint_twice_should_be_stale() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::new();
        let mut tracker = ScriptTracker::<u32>::default();

        let update = update_gen.create_update(
            vec![TxSpec {
                inputs: vec![Other(1_900)],
                outputs: vec![Mine(2_000, 0)],
                confirmed_at: Some(0),
                is_coinbase: false,
            }],
            0,
        );

        assert_eq!(tracker.apply_checkpoint(update.clone()), ApplyResult::Ok);
        assert_eq!(tracker.apply_checkpoint(update), ApplyResult::Stale);
    }
}
