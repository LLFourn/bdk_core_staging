use crate::{BlockTime, CheckPoint, HashMap, HashSet, PrevOuts};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use bitcoin::{
    self,
    hashes::{sha256, Hash, HashEngine},
    psbt::{self, PartiallySignedTransaction as Psbt},
    secp256k1::{Secp256k1, VerifyOnly},
    util::address::WitnessVersion,
    BlockHash, OutPoint, Script, Transaction, TxIn, TxOut, Txid,
};
use miniscript::{
    descriptor::DerivedDescriptorKey, psbt::PsbtInputExt, Descriptor, DescriptorPublicKey,
};

#[derive(Clone, Debug)]
pub struct DescriptorTracker {
    /// The descriptor we are tracking
    descriptor: Descriptor<DescriptorPublicKey>,
    /// Which txids are included in which checkpoints
    checkpointed_txs: BTreeMap<u32, _CheckPointData>,
    /// Index the outpoints owned by this tracker to the derivation index of script pubkey.
    txouts: BTreeMap<OutPoint, u32>,
    /// Which tx spent each output. This indexes the spends for every transaction regardless if the
    /// outpoint has one of our scripts pubkeys.
    spends: BTreeMap<OutPoint, (u32, Txid)>,
    /// The unspent txouts
    unspent: HashSet<OutPoint>,
    /// The ordered script pubkeys that have been derived from the descriptor.
    scripts: Vec<Script>,
    /// A reverse lookup from script to derivation index
    script_indexes: HashMap<Script, u32>,
    /// A lookup from script pubkey to outpoint
    script_txouts: BTreeMap<u32, HashSet<OutPoint>>,
    /// A set of script derivation indexes that haven't been spent to
    unused: BTreeSet<u32>,
    /// Map from txid to metadata
    txs: HashMap<Txid, AugmentedTx>,
    /// Index of transactions that are in the mempool
    mempool: HashSet<Txid>,
    // TODO change to blocktime + height
    // Optionally we need the consensus time i.e. Median time past
    // https://github.com/bitcoin/bitcoin/blob/a4e066af8573dcefb11dff120e1c09e8cf7f40c2/src/chain.h#L290-L302
    latest_blockheight: Option<u32>,
    /// The maximum number of checkpoints that the descriptor should store. When a new checkpoint is
    /// added which would push it above the limit we merege the oldest two checkpoints together.
    checkpoint_limit: usize,
    secp: Secp256k1<VerifyOnly>,
}

#[derive(Clone, Default)]
struct _CheckPointData {
    block_hash: BlockHash,
    ordered_txids: BTreeSet<(u32, Txid)>,
    accum_digest: sha256::HashEngine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckPointTxids {
    /// The TXIDs that are in this checkpoint. This implies the confirmation height of each
    /// transaction was less than or equal to this checkpoint's height but greater than the
    /// preceding checkpoint.
    pub txids: Vec<Txid>,
    /// We keep an digest of all transaction ids in this and all previous checkpoints.
    /// The TXIDs in the hash are ordered by confirmation time and then lexically.
    ///
    /// We keep this for two reaons:
    ///
    /// 1. If we have two checkpointed data sources that proport to follow the same descriptor we
    /// can tell quickly if they disagree and if so at which height do they disagree.
    /// 2. We want to be able to delete old checkpoints by merging their txids into a newer one.
    /// With this digest we can do that without changing the identity of the checkpoint that has
    /// the new txids merged into it.
    pub accum_digest: sha256::Hash,
}

impl From<_CheckPointData> for CheckPointTxids {
    fn from(from: _CheckPointData) -> Self {
        Self {
            txids: from
                .ordered_txids
                .into_iter()
                .map(|(_, txids)| txids)
                .collect(),
            accum_digest: sha256::Hash::from_engine(from.accum_digest),
        }
    }
}

impl core::fmt::Debug for _CheckPointData {
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

#[derive(Clone, Debug, PartialEq)]
pub enum UpdateResult {
    /// The update was applied successfully.
    // TODO: return a diff
    Ok,
    /// The update cannot be applied to the current state because it does not apply to the current
    /// tip of the tracker or does not invalidate the right checkpoint such that it does.
    Stale,
    /// The update you tried to apply was inconsistent with the current state.
    ///
    /// To resolve the consistency you can treat the update as invalid and ignore it or invalidate
    /// the checkpoint and re-apply it.
    Inconsistent {
        txid: Txid,
        conflicts_with: Txid,
        at_checkpoint: CheckPoint,
    },
}

impl DescriptorTracker {
    pub fn new(descriptor: Descriptor<DescriptorPublicKey>) -> Self {
        Self {
            descriptor,
            checkpointed_txs: Default::default(),
            secp: Secp256k1::verification_only(),
            txouts: Default::default(),
            spends: Default::default(),
            unspent: Default::default(),
            scripts: Default::default(),
            script_indexes: Default::default(),
            script_txouts: Default::default(),
            unused: Default::default(),
            txs: Default::default(),
            mempool: Default::default(),
            latest_blockheight: Default::default(),
            checkpoint_limit: usize::MAX,
        }
    }

    pub fn set_checkpoint_limit(&mut self, limit: usize) {
        assert!(limit > 0);
        self.checkpoint_limit = limit;
        self.apply_checkpoint_limit()
    }

    fn apply_checkpoint_limit(&mut self) {
        while self.checkpointed_txs.len() > self.checkpoint_limit {
            let first = *self.checkpointed_txs.iter().next().unwrap().0;
            self.merge_checkpoint(first);
        }
    }

    pub fn latest_blockheight(&self) -> Option<u32> {
        self.latest_blockheight
    }

    pub fn descriptor(&self) -> &Descriptor<DescriptorPublicKey> {
        &self.descriptor
    }

    pub fn next_derivation_index(&self) -> u32 {
        self.scripts.len() as u32
    }

    /// Get the transaction ids at that checkpoint
    ///
    /// As well as the list of transaction ids at the checkpoint (ordered in ascending confirmation
    /// height) [`CheckPointTxids`] includes a SHA256 hash over all the confirmed txids in the
    /// tracker up to that checkpoint.
    ///
    /// The `Txid`s are orderd first by their confirmation height (ascending) and then lexically by their `Txid`.
    ///
    /// ## Panics
    ///
    /// This will panic if the checkpoint doesn't exist.
    pub fn checkpoint_txids(&self, at: CheckPoint) -> CheckPointTxids {
        let data = self
            .checkpointed_txs
            .get(&at.height)
            .expect("the tracker did not have a checkpoint at that height");
        assert_eq!(
            data.block_hash, at.hash,
            "tracker had a different block hash for checkpoint at that height"
        );
        data.clone().into()
    }

    pub fn latest_checkpoint(&self) -> Option<CheckPoint> {
        self.checkpointed_txs
            .iter()
            .last()
            .map(|(height, data)| CheckPoint {
                height: *height,
                hash: data.block_hash,
            })
    }

    pub fn checkpoint_at(&self, height: u32) -> Option<CheckPoint> {
        let data = self.checkpointed_txs.get(&height)?;
        Some(CheckPoint {
            height,
            hash: data.block_hash,
        })
    }

    fn best_checkpoint_for(&self, height: u32) -> Option<CheckPoint> {
        let (cp_height, data) = self.checkpointed_txs.range(height..).next()?;

        Some(CheckPoint {
            height: *cp_height,
            hash: data.block_hash,
        })
    }

    /// Iterate checkpoints from newest to oldtest
    pub fn iter_checkpoints(&self) -> impl Iterator<Item = CheckPoint> + '_ {
        self.checkpointed_txs
            .iter()
            .rev()
            .map(|(height, data)| CheckPoint {
                height: *height,
                hash: data.block_hash,
            })
    }

    fn remove_tx(&mut self, txid: Txid) {
        let aug_tx = match self.txs.remove(&txid) {
            Some(aug_tx) => aug_tx,
            None => {
                debug_assert!(!self.mempool.contains(&txid));
                return;
            }
        };
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
            }
        }

        self.mempool.remove(&txid);
    }

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
            PrevOuts::Spend(txouts) => txouts.iter().map(|input| input.value).sum(),
        };
        let outputs_sum: u64 = tx.output.iter().map(|out| out.value).sum();
        // we need to saturating sub since we want coinbase txs to map to 0 fee and
        // this subtraction will be negative for coinbase txs.
        let fee = inputs_sum.saturating_sub(outputs_sum);
        let feerate = fee as f32 / tx.weight() as f32;

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
            // the only things we conflict with are in the mempool and this is confirmed so delete it
            for conflicting_txid in conflicts {
                self.remove_tx(conflicting_txid);
            }
        } else {
            // NOTE: We have already made sure that all conflicts are unconfirmed.
            // TODO: Make resolution for mempool conflicts customizable
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

        self.txs.insert(
            txid,
            AugmentedTx {
                tx,
                fee,
                feerate,
                confirmation_time,
            },
        );

        match confirmation_time {
            Some(confirmation_time) => {
                // Find the first checkpoint above or equal to the tx's height
                let (checkpoint_height, checkpoint_data) = self
                    .checkpointed_txs
                    .range_mut(confirmation_time.height..)
                    .next()
                    .expect("the caller must have checked that no txs are outside of range");

                if checkpoint_data
                    .ordered_txids
                    .insert((confirmation_time.height, txid))
                {
                    // if we modify the checkpoint return the height we modified
                    Some(*checkpoint_height)
                } else {
                    None
                }
            }
            None => {
                self.mempool.insert(txid);
                None
            }
        }
    }

    fn invalidate_checkpoint(&mut self, height: u32) {
        let removed = self.checkpointed_txs.split_off(&height);
        let txs_to_remove = removed
            .values()
            .rev()
            .map(|data| data.ordered_txids.iter().map(|(_, txid)| txid))
            .flatten();
        for tx_to_remove in txs_to_remove {
            self.remove_tx(*tx_to_remove);
        }
    }

    pub fn apply_update(&mut self, update: Update) -> UpdateResult {
        // Do consistency checks first so we don't mutate anything until we're sure the update is
        // valid.
        for (_, tx, confirmation_time) in &update.transactions {
            let txid = tx.txid();
            if let Some(existing) = self.txs.get(&tx.txid()) {
                if let Some(existing_time) = existing.confirmation_time {
                    if confirmation_time != &Some(existing_time) {
                        let at_checkpoint = self
                            .best_checkpoint_for(existing_time.height)
                            .expect("must exist since there's a confirmed tx");
                        return UpdateResult::Inconsistent {
                            txid,
                            conflicts_with: existing.tx.txid(),
                            at_checkpoint,
                        };
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
                    let at_checkpoint = self
                        .best_checkpoint_for(conflicting_conftime.height)
                        .expect("must exist since there's a confirmed tx");
                    return UpdateResult::Inconsistent {
                        txid,
                        conflicts_with: *conflicting_txid,
                        at_checkpoint,
                    };
                }
            }
        }

        if let Some(last_active_index) = update.last_active_index {
            // It's possible that we find a script derived at a higher index than what we have given
            // out in the case where another system is deriving from the same descriptor.
            self.derive_scripts(last_active_index);
        }

        // look for invalidated and check that start tip is the one before it.
        match update.invalidate {
            Some(checkpoint_reset) => match self.checkpointed_txs.get(&checkpoint_reset.height) {
                Some(checkpoint_data) => {
                    if checkpoint_data.block_hash != checkpoint_reset.hash {
                        if self
                            .checkpointed_txs
                            .range(..checkpoint_reset.height)
                            .last()
                            .map(|(height, data)| CheckPoint {
                                height: *height,
                                hash: data.block_hash,
                            })
                            == update.base_tip
                        {
                            self.invalidate_checkpoint(checkpoint_reset.height);
                        } else {
                            return UpdateResult::Stale;
                        }
                    } else {
                        return UpdateResult::Stale;
                    }
                }
                None => return UpdateResult::Stale,
            },
            None => {
                if update.base_tip != self.latest_checkpoint() {
                    return UpdateResult::Stale;
                }
            }
        }

        if update.mempool_is_total_set {
            // This update will include everything in the mempool that is relevent to the tracker so
            // we clear everything.
            self.clear_mempool();
        }

        // Insert a new empty checkpoint at the update height
        self.checkpointed_txs
            .entry(update.new_tip.height)
            .or_insert_with(|| _CheckPointData {
                block_hash: update.new_tip.hash,
                ..Default::default()
            });

        let mut deepest_change = None;
        for (vouts, tx, confirmation_time) in update.transactions {
            if let Some(change) = self.add_tx(vouts, tx, confirmation_time) {
                deepest_change = Some(deepest_change.unwrap_or(0).min(change));
            }
        }

        let txids_in_new_checkpoint = &self
            .checkpointed_txs
            .values()
            .rev()
            .next()
            .unwrap()
            .ordered_txids;

        if txids_in_new_checkpoint.is_empty() {
            // the new checkpoint we inserted ends up empty so delete it
            self.checkpointed_txs.remove(&update.new_tip.height);
        }

        if let Some(change_depth) = deepest_change {
            self.recompute_txid_digests(change_depth);
        }

        self.latest_blockheight = Some(update.new_tip.height);
        self.apply_checkpoint_limit();

        debug_assert!(self.is_latest_checkpoint_hash_correct());

        UpdateResult::Ok
    }

    fn recompute_txid_digests(&mut self, from: u32) {
        let mut prev_accum_digest = self
            .checkpointed_txs
            .range(..from)
            .next()
            .map(|(_, prev)| prev.accum_digest.clone())
            .unwrap_or_else(sha256::HashEngine::default);

        for (_height, data) in self.checkpointed_txs.range_mut(from..) {
            let mut accum_digest = prev_accum_digest.clone();
            for (_, txid) in &data.ordered_txids {
                accum_digest.input(txid);
            }
            data.accum_digest = accum_digest.clone();
            prev_accum_digest = accum_digest;
        }
    }

    /// Takes the checkpoint at index and merges its transactions into the next checkpoint
    pub fn merge_checkpoint(&mut self, height: u32) {
        if let Some(checkpoint) = self.checkpointed_txs.remove(&height) {
            match self.checkpointed_txs.range_mut((height + 1)..).next() {
                Some((_, next_one)) => next_one.ordered_txids.extend(checkpoint.ordered_txids),
                // put it back
                None => {
                    self.checkpointed_txs.insert(height, checkpoint);
                }
            }
        }

        debug_assert!(self.is_latest_checkpoint_hash_correct());
    }

    pub fn clear_mempool(&mut self) {
        let mempool = core::mem::replace(&mut self.mempool, Default::default());
        for txid in mempool {
            self.remove_tx(txid);
        }

        debug_assert!(self.mempool.is_empty())
    }

    pub fn disconnect_block(&mut self, block_height: u32, block_hash: BlockHash) {
        // Can't guarantee that mempool is consistent with chain after we disconnect a block so we
        // clear it.
        // TODO: it would be nice if we could only delete those transactions that are
        // inconsistent by recording the latest block they were included in.
        self.clear_mempool();
        if let Some(checkpoint_data) = self.checkpointed_txs.get(&block_height) {
            if checkpoint_data.block_hash == block_hash {
                self.invalidate_checkpoint(block_height);
            }
        }
    }

    /// Iterate over transactions.
    ///
    /// This iterates transactions in the mempool first and then the rest are ordered starting from
    /// most recently confirmed.
    pub fn iter_tx(&self) -> impl Iterator<Item = (Txid, &AugmentedTx)> {
        self.txs.iter().map(|(txid, tx)| (*txid, tx))
    }

    pub fn iter_tx_by_age(&self) -> impl DoubleEndedIterator<Item = (Txid, &AugmentedTx)> + '_ {
        let mempool_tx = self
            .mempool
            .iter()
            .map(|txid| (*txid, self.txs.get(txid).unwrap()));
        let confirmed_tx = self.checkpointed_txs.iter().rev().flat_map(|(_, data)| {
            data.ordered_txids
                .iter()
                .map(|(_, txid)| (*txid, self.txs.get(txid).unwrap()))
        });

        mempool_tx.chain(confirmed_tx)
    }

    pub fn iter_unspent(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.unspent
            .iter()
            .map(|txo| (txo, self.txouts.get(txo).expect("txout must exist")))
            .map(|(txo, index)| self.create_txout(*txo, *index))
    }

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

    pub fn iter_txout(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.txouts
            .iter()
            .map(|(outpoint, data)| self.create_txout(*outpoint, *data))
    }

    pub fn get_txout(&self, txo: OutPoint) -> Option<LocalTxOut> {
        let data = self.txouts.get(&txo)?;
        Some(self.create_txout(txo, *data))
    }

    pub fn get_tx(&self, txid: Txid) -> Option<&AugmentedTx> {
        self.txs.get(&txid)
    }

    /// Iterates over all the script pubkeys of a descriptor.
    ///
    /// This method does **not** use the tracker's stored scripts and returned iterator does not
    /// hold a reference to the tracker. This allows it to be sent between threads. If the
    /// descriptor `is_deriveable` then the iterator will derive and emit all non-hardened indexes
    /// of the descriptor otherwise it will just have one script in it.
    ///
    /// **WARNING**: never turn these into addresses or send coins to them.
    /// The tracker may not be able to find them.
    /// To get a script you can use as an address use [`derive_next`].
    ///
    /// [`derive_next`]: Self::derive_next
    pub fn iter_all_scripts(&self) -> impl Iterator<Item = Script> + Send {
        let descriptor = self.descriptor.clone();
        let end = if self.descriptor.is_deriveable() {
            // Because we only iterate over non-hardened indexes there are 2^31 values
            (1 << 31) - 1
        } else {
            0
        };

        let secp = self.secp.clone();
        (0..=end).map(move |i| {
            descriptor
                .derive(i)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey()
        })
    }

    /// Returns the script that has been derived at the index.
    ///
    /// If that index hasn't been derived yet it will return None instead.
    pub fn script_at_index(&self, index: u32) -> Option<&Script> {
        self.scripts.get(index as usize)
    }

    /// Derives a new script pubkey which can be turned into an address.
    ///
    /// The tracker returns a new address for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn derive_new(&mut self) -> (u32, &Script) {
        let next_derivation_index = if self.descriptor.is_deriveable() {
            0
        } else {
            self.scripts.len() as u32
        };
        self.derive_scripts(next_derivation_index);
        let script = self
            .scripts
            .get(next_derivation_index as usize)
            .expect("we just derived to that index");
        (next_derivation_index, script)
    }

    /// Derives and stores a new address only if we haven't already got one that hasn't been used
    /// yet.
    pub fn derive_next_unused(&mut self) -> (u32, &Script) {
        let need_new = self.iter_unused_scripts().next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new()
        } else {
            self.iter_unused_scripts().next().unwrap()
        }
    }

    /// Iterate over the scripts that have been derived already
    pub fn iter_scripts(&self) -> impl Iterator<Item = &Script> {
        self.scripts.iter()
    }

    /// Iterate over the scripts that have been derived but we have not seen a transaction spending
    /// from it.
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

    /// Derives and stores all the scripts **up to and including** `end`.
    ///
    /// Returns whether any new scripts needed deriving.
    pub fn derive_scripts(&mut self, end: u32) -> bool {
        let end = match self.descriptor.is_deriveable() {
            false => 0,
            true => end,
        };

        let needed = (end + 1).saturating_sub(self.scripts.len() as u32);
        for index in self.scripts.len()..self.scripts.len() + needed as usize {
            let script = self
                .descriptor
                .derive(index as u32)
                .derived_descriptor(&self.secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            self.scripts.push(script.clone());
            self.script_indexes.insert(script.clone(), index as u32);
            self.unused.insert(index as u32);
        }

        needed == 0
    }

    /// Returns at what derivation index a script pubkey was derived at.
    pub fn index_of_derived_script(&self, script: &Script) -> Option<u32> {
        self.script_indexes.get(script).cloned()
    }

    /// The maximum satisfaction weight of a descriptor
    pub fn max_satisfaction_weight(&self) -> u32 {
        self.descriptor
            .derive(0)
            .max_satisfaction_weight()
            .expect("descriptor is well formed") as u32
    }

    /// The dust value for any script used as a script pubkey on the network.
    ///
    /// Transactions with output containing script pubkeys from this descriptor with values below
    /// this will not be relayed by the network.
    pub fn dust_value(&self) -> u64 {
        self.descriptor
            .derive(0)
            .script_pubkey()
            .dust_value()
            .as_sat()
    }

    /// Prepare an input for insertion into a PSBT
    pub fn prime_input(&self, op: OutPoint) -> Option<PrimedInput> {
        let derivation_index = self.txouts.get(&op)?;
        let descriptor = self.descriptor().derive(*derivation_index);
        let mut psbt_input = psbt::Input::default();

        let prev_tx = self
            .txs
            .get(&op.txid)
            .expect("since the txout exists so mus the transaction");

        match self.descriptor().desc_type().segwit_version() {
            Some(version) => {
                if version < WitnessVersion::V1 {
                    psbt_input.non_witness_utxo = Some(prev_tx.tx.clone());
                }
                psbt_input.witness_utxo = Some(prev_tx.tx.output[op.vout as usize].clone());
            }
            None => psbt_input.non_witness_utxo = Some(prev_tx.tx.clone()),
        }

        psbt_input
            .update_with_descriptor_unchecked(&descriptor)
            .expect("conversion error cannot happen if descriptor is well formed");

        let primed_input = PrimedInput {
            descriptor,
            psbt_input,
        };

        Some(primed_input)
    }

    /// internal debug function to double check correctness of the accumulated digest
    fn is_latest_checkpoint_hash_correct(&self) -> bool {
        if let Some(tip) = self.latest_checkpoint() {
            let tip_txids = self.checkpoint_txids(tip);
            let mut txs = self
                .iter_tx()
                .filter(|(_, tx)| tx.confirmation_time.is_some())
                .collect::<Vec<_>>();
            txs.sort_by_key(|(_, tx)| tx.confirmation_time.unwrap().height);
            let mut hasher = sha256::HashEngine::default();
            for (txid, _) in txs {
                hasher.input(&txid);
            }
            let txid_hash = sha256::Hash::from_engine(hasher);
            txid_hash == tip_txids.accum_digest
        } else {
            true
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Update {
    pub transactions: Vec<(PrevOuts, Transaction, Option<BlockTime>)>,
    pub mempool_is_total_set: bool,
    pub last_active_index: Option<u32>,
    /// The data in the update can be applied upon this checkpoint. If None then it is not
    /// consistent with any particular tip (apart from new tip) and so should form the base
    pub base_tip: Option<CheckPoint>,
    /// Invalidates a particular checkpoint
    pub invalidate: Option<CheckPoint>,
    /// The data is valid with respect to new_tip
    pub new_tip: CheckPoint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedTx {
    pub tx: Transaction,
    pub fee: u64,
    pub feerate: f32,
    pub confirmation_time: Option<BlockTime>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTxOut {
    pub value: u64,
    pub spent_by: Option<(u32, Txid)>,
    pub outpoint: OutPoint,
    pub derivation_index: u32,
    pub confirmed_at: Option<BlockTime>,
}

pub trait MultiTracker {
    fn iter_unspent(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_>;
    fn iter_txout(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_>;
    fn latest_blockheight(&self) -> Option<u32>;
    fn create_psbt<I, O>(
        &self,
        inputs: I,
        outputs: O,
    ) -> (Psbt, BTreeMap<usize, Descriptor<DerivedDescriptorKey>>)
    where
        I: IntoIterator<Item = OutPoint>,
        O: IntoIterator<Item = TxOut>;
}

impl<'a> MultiTracker for [DescriptorTracker] {
    fn iter_unspent(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_> {
        Box::new(
            self.into_iter()
                .enumerate()
                .flat_map(|(i, tracker)| tracker.iter_unspent().map(move |u| (i, u))),
        )
    }

    fn iter_txout(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_> {
        Box::new(
            self.into_iter()
                .enumerate()
                .flat_map(|(i, tracker)| tracker.iter_txout().map(move |u| (i, u))),
        )
    }

    fn latest_blockheight(&self) -> Option<u32> {
        self.into_iter()
            .filter_map(|tracker| tracker.latest_blockheight())
            .max()
    }

    fn create_psbt<I, O>(
        &self,
        inputs: I,
        outputs: O,
    ) -> (Psbt, BTreeMap<usize, Descriptor<DerivedDescriptorKey>>)
    where
        I: IntoIterator<Item = OutPoint>,
        O: IntoIterator<Item = TxOut>,
    {
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

        let mut descriptors = BTreeMap::new();

        let mut psbt = Psbt::from_unsigned_tx(unsigned_tx).unwrap();

        for ((input_index, psbt_input), txin) in psbt
            .inputs
            .iter_mut()
            .enumerate()
            .zip(&psbt.unsigned_tx.input)
        {
            if let Some(primed_input) = self
                .iter()
                .find_map(|tracker| tracker.prime_input(txin.previous_output))
            {
                *psbt_input = primed_input.psbt_input;
                descriptors.insert(input_index, primed_input.descriptor);
            }
        }

        (psbt, descriptors)
    }
}

#[derive(Debug, Clone)]
pub struct PrimedInput {
    pub descriptor: Descriptor<DerivedDescriptorKey>,
    pub psbt_input: psbt::Input,
}

#[cfg(test)]
mod test {
    use bitcoin::{BlockHash, Transaction, TxIn, TxOut};
    use core::cmp::max;

    use super::*;

    const DESCRIPTOR: &'static str = "wpkh(xpub6ERApfZwUNrhLCkDtcHTcxd75RbzS1ed54G1LkBUHQVHQKqhMkhgbmJbZRkrgZw4koxb5JaHWkY4ALHY2grBGRjaDMzQLcgJvLJuZZvRcEL)";

    pub enum IOSpec {
        Mine(u64, usize),
        Other(u64),
    }

    pub struct TxSpec {
        inputs: Vec<IOSpec>,
        outputs: Vec<IOSpec>,
        confirmed_at: Option<u32>,
        is_coinbase: bool,
    }

    #[derive(Default, Clone, Debug)]
    struct UpdateGen {
        vout_counter: u32,
        prev_tip: Option<CheckPoint>,
    }

    impl UpdateGen {
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
            descriptor: &Descriptor<DescriptorPublicKey>,
            txs: Vec<TxSpec>,
            checkpoint_height: u32,
        ) -> Update {
            let secp = Secp256k1::verification_only();
            let last_active_index = txs.iter().fold(None, |lai, tx_spec| {
                tx_spec
                    .inputs
                    .iter()
                    .chain(tx_spec.outputs.iter())
                    .fold(lai, |lai, spec| match (lai, spec) {
                        (Some(lai), IOSpec::Mine(_, index)) => Some(max(*index as u32, lai)),
                        (None, IOSpec::Mine(_, index)) => Some(*index as u32),
                        _ => lai,
                    })
            });
            let transactions = txs
                .into_iter()
                .map(|tx_spec| {
                    (
                        match tx_spec.is_coinbase {
                            false => PrevOuts::Spend(
                                tx_spec
                                    .inputs
                                    .iter()
                                    .map(|in_spec| match in_spec {
                                        IOSpec::Mine(value, index) => TxOut {
                                            value: *value,
                                            script_pubkey: descriptor
                                                .derive(*index as u32)
                                                .derived_descriptor(&secp)
                                                .unwrap()
                                                .script_pubkey(),
                                        },
                                        IOSpec::Other(value) => TxOut {
                                            value: *value,
                                            script_pubkey: Default::default(),
                                        },
                                    })
                                    .collect(),
                            ),
                            true => {
                                todo!()
                            }
                        },
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
                                        script_pubkey: descriptor
                                            .derive(index as u32)
                                            .derived_descriptor(&secp)
                                            .unwrap()
                                            .script_pubkey(),
                                    },
                                })
                                .collect(),
                        },
                        tx_spec.confirmed_at.map(|confirmed_at| BlockTime {
                            height: confirmed_at,
                            time: confirmed_at as u64,
                        }),
                    )
                })
                .collect();

            let new_tip = CheckPoint {
                height: checkpoint_height,
                hash: BlockHash::default(),
            };

            let update = Update {
                transactions,
                last_active_index,
                new_tip,
                invalidate: None,
                mempool_is_total_set: true,
                base_tip: self.prev_tip,
            };

            self.prev_tip = Some(new_tip);

            update
        }
    }

    #[test]
    fn no_checkpoint_and_then_confirm() {
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());
        use IOSpec::*;

        let mut update = update_gen.create_update(
            tracker.descriptor(),
            vec![TxSpec {
                inputs: vec![Other(2_000)],
                outputs: vec![Mine(1_000, 0), Other(1_800)],
                confirmed_at: None,
                is_coinbase: false,
            }],
            0,
        );

        assert_eq!(tracker.apply_update(update.clone()), UpdateResult::Ok);

        let txouts = tracker.iter_txout().collect::<Vec<_>>();
        let txs = tracker.iter_tx().collect::<Vec<_>>();
        let unspent = tracker.iter_unspent().collect::<Vec<_>>();
        let checkpoints = tracker.iter_checkpoints().collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        assert_eq!(unspent, txouts);
        assert_eq!(txs.len(), 1);
        assert_eq!(checkpoints.len(), 0);
        assert_eq!(txouts.len(), 1);

        update.transactions[0].2 = Some(BlockTime { height: 1, time: 1 });
        update.new_tip = CheckPoint {
            height: update.new_tip.height + 1,
            hash: update.new_tip.hash,
        };

        assert_eq!(tracker.apply_update(update), UpdateResult::Ok);

        let txs = tracker.iter_tx().collect::<Vec<_>>();
        let checkpoints = tracker.iter_checkpoints().collect::<Vec<_>>();
        let txouts = tracker.iter_txout().collect::<Vec<_>>();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(txouts.len(), 1);
        assert_eq!(
            tracker.checkpoint_txids(checkpoints[0]).txids,
            txs.iter().map(|(x, _)| *x).collect::<Vec<_>>()
        );
        assert!(tracker.is_latest_checkpoint_hash_correct());
    }

    #[test]
    fn two_checkpoints_then_merege() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());

        assert_eq!(
            tracker.apply_update(update_gen.create_update(
                tracker.descriptor(),
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
            UpdateResult::Ok
        );

        assert_eq!(
            tracker.apply_update(update_gen.create_update(
                tracker.descriptor(),
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
            UpdateResult::Ok
        );

        assert_eq!(tracker.iter_txout().count(), 4);

        // there is no checkpoint here
        tracker.merge_checkpoint(0);
        assert_eq!(tracker.iter_checkpoints().count(), 2);

        tracker.merge_checkpoint(1);
        assert_eq!(tracker.iter_checkpoints().count(), 1);

        let txdata = tracker.checkpoint_txids(tracker.checkpoint_at(3).unwrap());
        assert_eq!(txdata.txids.len(), 4);
    }

    #[test]
    fn out_of_order_tx_is_before_first_checkpoint() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());

        assert_eq!(
            tracker.apply_update(update_gen.create_update(
                tracker.descriptor(),
                vec![TxSpec {
                    inputs: vec![Other(2_000)],
                    outputs: vec![Mine(2_000, 0)],
                    confirmed_at: Some(1),
                    is_coinbase: false,
                },],
                1,
            )),
            UpdateResult::Ok
        );

        assert_eq!(
            tracker.apply_update(update_gen.create_update(
                tracker.descriptor(),
                vec![TxSpec {
                    inputs: vec![Other(2_000)],
                    outputs: vec![Mine(2_000, 1)],
                    confirmed_at: Some(0),
                    is_coinbase: false,
                },],
                2,
            )),
            UpdateResult::Ok
        );

        assert!(tracker.is_latest_checkpoint_hash_correct());
    }

    #[test]
    fn checkpoint_limit_is_applied() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());
        tracker.set_checkpoint_limit(5);

        for i in 0..10 {
            assert_eq!(
                tracker.apply_update(update_gen.create_update(
                    tracker.descriptor(),
                    vec![TxSpec {
                        inputs: vec![Other(2_000)],
                        outputs: vec![Mine(2_000, i)],
                        confirmed_at: Some(i as u32),
                        is_coinbase: false,
                    },],
                    i as u32,
                )),
                UpdateResult::Ok
            );
        }

        assert_eq!(tracker.iter_tx().count(), 10);
        assert_eq!(tracker.iter_checkpoints().count(), 5);
    }
}
