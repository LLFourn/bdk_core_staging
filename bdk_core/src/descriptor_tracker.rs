use crate::{collections::*, AugmentedTx, BlockId, PrimedInput};
use bitcoin::{
    self,
    hashes::sha256,
    psbt::{self, PartiallySignedTransaction as Psbt},
    secp256k1::{Secp256k1, VerifyOnly},
    util::address::WitnessVersion,
    OutPoint, Script, Transaction, TxIn, TxOut, Txid,
};
use miniscript::{
    descriptor::DefiniteDescriptorKey, psbt::PsbtInputExt, Descriptor, DescriptorPublicKey,
};

use crate::{script_tracker::ScriptTracker, ApplyResult, Box, CheckpointCandidate, LocalTxOut};

#[derive(Clone, Debug)]
pub struct DescriptorTracker {
    /// The descriptor we are tracking
    descriptor: Descriptor<DescriptorPublicKey>,
    inner: ScriptTracker,
    secp: Secp256k1<VerifyOnly>,
}

impl DescriptorTracker {
    pub fn new(descriptor: Descriptor<DescriptorPublicKey>) -> Self {
        Self {
            inner: ScriptTracker::default(),
            secp: Secp256k1::verification_only(),
            descriptor,
        }
    }

    pub fn descriptor(&self) -> &Descriptor<DescriptorPublicKey> {
        &self.descriptor
    }

    /// Get the next index of underived script pubkey from the descriptor
    pub fn next_derivation_index(&self) -> u32 {
        self.inner.scripts().len() as u32
    }

    /// Derives and stores a new scriptpubkey only if we haven't already got one that hasn't received any
    /// coins yet.
    pub fn derive_next_unused(&mut self) -> (u32, &Script) {
        let need_new = self.inner.iter_unused_scripts().next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new()
        } else {
            self.inner.iter_unused_scripts().next().unwrap()
        }
    }

    /// Derives a new script pubkey which can be turned into an address.
    ///
    /// The tracker returns a new address for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn derive_new(&mut self) -> (u32, &Script) {
        let next_derivation_index = if self.descriptor.has_wildcard() {
            self.scripts().len() as u32
        } else {
            0
        };
        self.derive_scripts(next_derivation_index);
        let script = self
            .scripts()
            .get(next_derivation_index as usize)
            .expect("we just derived to that index");
        (self.scripts().len() as u32, script)
    }

    pub fn scripts(&self) -> &[Script] {
        self.inner.scripts()
    }

    /// Derives and stores all the scripts **up to and including** `end`.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn derive_scripts(&mut self, end: u32) -> bool {
        let end = match self.descriptor.has_wildcard() {
            false => 0,
            true => end,
        };

        let needed = (end + 1).saturating_sub(self.scripts().len() as u32);
        for index in self.scripts().len()..self.scripts().len() + needed as usize {
            let script = self
                .descriptor
                .at_derivation_index(index as u32)
                .derived_descriptor(&self.secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            let _index = self.inner.add_script(script.clone());
            assert_eq!(index, _index);
        }

        needed == 0
    }

    /// Iterates over all the script pubkeys of a descriptor.
    ///
    /// This method does **not** use the tracker's stored scripts and returned iterator does not
    /// hold a reference to the tracker. This allows it to be sent between threads. If the
    /// descriptor `has_wildcard` then the iterator will derive and emit all non-hardened indexes
    /// of the descriptor otherwise it will just have one script in it.
    ///
    /// **WARNING**: never turn these into addresses or send coins to them.
    /// The tracker may not be able to find them.
    /// To get a script you can use as an address use [`derive_next`].
    ///
    /// [`derive_next`]: Self::derive_next
    pub fn iter_all_scripts(&self) -> impl Iterator<Item = Script> + Send {
        let descriptor = self.descriptor.clone();
        let end = if self.descriptor.has_wildcard() {
            // Because we only iterate over non-hardened indexes there are 2^31 values
            (1 << 31) - 1
        } else {
            0
        };

        let secp = self.secp.clone();
        (0..=end).map(move |i| {
            descriptor
                .at_derivation_index(i)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey()
        })
    }

    pub fn apply_checkpoint(&mut self, new_checkpoint: CheckpointCandidate) -> ApplyResult {
        // Derive all scripts up to the last active one so we find all the txos owned by this
        // tracker.
        if let Some(last_active_index) = new_checkpoint.last_active_index {
            self.derive_scripts(last_active_index);
        }
        self.inner.apply_checkpoint(new_checkpoint)
    }

    /// The maximum satisfaction weight of a descriptor
    pub fn max_satisfaction_weight(&self) -> u32 {
        self.descriptor
            .at_derivation_index(0)
            .max_satisfaction_weight()
            .expect("descriptor is well formed") as u32
    }

    /// The dust value for any script used as a script pubkey on the network.
    ///
    /// Transactions with output containing script pubkeys from this descriptor with values below
    /// this will not be relayed by the network.
    pub fn dust_value(&self) -> u64 {
        self.descriptor
            .at_derivation_index(0)
            .script_pubkey()
            .dust_value()
            .as_sat()
    }

    /// Prepare an input for insertion into a PSBT
    pub fn prime_input(&self, op: OutPoint) -> Option<PrimedInput> {
        let derivation_index = self.inner.get_txout(op)?.derivation_index;
        let descriptor = self.descriptor().at_derivation_index(derivation_index);
        let mut psbt_input = psbt::Input::default();

        let prev_tx = self
            .inner
            .get_tx(op.txid)
            .expect("since the txout exists so must the transaction");

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

    /// Return an immutable reference to the internal script tracker
    pub fn inner(&self) -> &ScriptTracker {
        &self.inner
    }

    /// Iterate over all transactions in our transaction store.
    /// Can be both related/unrelated and/or confirmed/unconfirmed.
    pub fn iter_tx(&self) -> impl Iterator<Item = (Txid, &AugmentedTx)> {
        self.inner.iter_tx()
    }

    /// Iterates over all transactions related to the descriptor ordered by decending confirmation
    /// with those transactions that are unconfirmed first.
    ///
    /// "related" means that the transactoin has an output with a script pubkey produced by the
    /// descriptor or it spends from such an output.
    pub fn iter_tx_by_confirmation_time(
        &self,
    ) -> impl DoubleEndedIterator<Item = (Txid, &AugmentedTx)> + '_ {
        self.inner.iter_tx_by_confirmation_time()
    }

    /// Iterate over unspent [LocalTxOut]s
    pub fn iter_unspent(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.inner.iter_unspent()
    }

    /// Iterate over all the transaction outputs discovered by the tracker with script pubkeys
    /// matches those stored by the tracker.
    pub fn iter_txout(&self) -> impl Iterator<Item = LocalTxOut> + '_ {
        self.inner.iter_txout()
    }

    /// Return an iterator over [BlockId] from newest to oldest, for this tracker
    pub fn iter_checkpoints(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.inner.iter_checkpoints()
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
        self.inner.checkpoint_txids(checkpoint_id)
    }

    /// Gets the SHA256 hash of all the `Txid`s of all the transactions included in all checkpoints
    /// up to and including `checkpoint_id`.
    ///
    /// ## Panics
    ///
    /// This will panic if a checkpoint doesn't exist with `checkpoint_id`
    pub fn accum_digest_at(&self, checkpoint_id: BlockId) -> sha256::Hash {
        self.inner.accum_digest_at(checkpoint_id)
    }

    /// Get the [BlockId] for the last known tip.
    pub fn latest_checkpoint(&self) -> Option<BlockId> {
        self.inner.latest_checkpoint()
    }

    /// Get the checkpoint id at the given height if it exists
    pub fn checkpoint_at(&self, height: u32) -> Option<BlockId> {
        self.inner.checkpoint_at(height)
    }

    /// Takes the checkpoint at a height and merges its transactions into the next checkpoint
    pub fn merge_checkpoint(&mut self, height: u32) {
        self.inner.merge_checkpoint(height)
    }

    /// Set the checkpoint limit for this tracker.
    /// If the limit is exceeded the last two checkpoints are merged together.
    pub fn set_checkpoint_limit(&mut self, limit: usize) {
        self.inner.set_checkpoint_limit(limit)
    }
}

impl AsRef<ScriptTracker> for DescriptorTracker {
    fn as_ref(&self) -> &ScriptTracker {
        &self.inner
    }
}

/// A trait implementing multiple descriptor tracker.
pub trait MultiTracker {
    fn iter_unspent(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_>;
    fn iter_txout(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_>;
    fn create_psbt<I, O>(
        &self,
        inputs: I,
        outputs: O,
    ) -> (Psbt, BTreeMap<usize, Descriptor<DefiniteDescriptorKey>>)
    where
        I: IntoIterator<Item = OutPoint>,
        O: IntoIterator<Item = TxOut>;
}

impl<'a> MultiTracker for [DescriptorTracker] {
    fn iter_unspent(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_> {
        Box::new(
            self.into_iter()
                .enumerate()
                .flat_map(|(i, tracker)| tracker.inner().iter_unspent().map(move |u| (i, u))),
        )
    }

    fn iter_txout(&self) -> Box<dyn Iterator<Item = (usize, LocalTxOut)> + '_> {
        Box::new(
            self.into_iter()
                .enumerate()
                .flat_map(|(i, tracker)| tracker.inner().iter_txout().map(move |u| (i, u))),
        )
    }

    fn create_psbt<I, O>(
        &self,
        inputs: I,
        outputs: O,
    ) -> (Psbt, BTreeMap<usize, Descriptor<DefiniteDescriptorKey>>)
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

#[cfg(test)]
mod test {
    use crate::{BlockId, BlockTime, PrevOuts, Vec};
    use bitcoin::{BlockHash, OutPoint, Transaction, TxIn, TxOut, Txid};
    use core::cmp::max;

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

    #[derive(Default, Clone, Debug)]
    struct UpdateGen {
        vout_counter: u32,
        prev_tip: Option<BlockId>,
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
        ) -> CheckpointCandidate {
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
                                                .at_derivation_index(*index as u32)
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
                            time: confirmed_at as u64,
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
                last_active_index,
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
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());
        use IOSpec::*;

        let mut checkpoint = update_gen.create_update(
            tracker.descriptor(),
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

        checkpoint.transactions[0].2 = Some(BlockTime { height: 1, time: 1 });
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
    }

    #[test]
    fn two_checkpoints_then_merege() {
        use IOSpec::*;
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
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
            ApplyResult::Ok
        );

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
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
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                tracker.descriptor(),
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
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                tracker.descriptor(),
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

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(
                tracker.descriptor(),
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
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());
        tracker.set_checkpoint_limit(5);

        for i in 0..10 {
            assert_eq!(
                tracker.apply_checkpoint(update_gen.create_update(
                    tracker.descriptor(),
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
        let mut update_gen = UpdateGen::default();
        let mut tracker = DescriptorTracker::new(DESCRIPTOR.parse().unwrap());
        let txs = (0..100)
            .map(|_| TxSpec {
                inputs: vec![Other(1_900)],
                outputs: vec![Mine(2_000, 0)],
                confirmed_at: Some(1),
                is_coinbase: false,
            })
            .collect();

        assert_eq!(
            tracker.apply_checkpoint(update_gen.create_update(tracker.descriptor(), txs, 1,)),
            ApplyResult::Ok
        );
    }
}
