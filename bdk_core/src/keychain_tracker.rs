use crate::collections::*;
use crate::SpkTracker;
use bitcoin::secp256k1::Secp256k1;
use bitcoin::util::address::WitnessVersion;
use bitcoin::OutPoint;
use bitcoin::Transaction;
use bitcoin::TxIn;
use bitcoin::TxOut;
use bitcoin::{util::psbt::PartiallySignedTransaction as Psbt, Script};
use core::ops::Deref;
use core::ops::DerefMut;
use miniscript::psbt::PsbtInputExt;
use miniscript::DefiniteDescriptorKey;
use miniscript::Descriptor;
use miniscript::DescriptorPublicKey;

/// A convienient way of tracking script pubkeys associated with one or more descriptors together.
///
/// `DeRef`s to the inner [`SpkTracker`]
///
/// [`SpkTracker`]: crate::SpkTracker
#[derive(Clone, Debug)]
pub struct KeychainTracker<K> {
    inner: SpkTracker<(K, u32)>,
    descriptors: BTreeMap<K, Descriptor<DescriptorPublicKey>>,
}

impl<K> Default for KeychainTracker<K> {
    fn default() -> Self {
        Self {
            inner: SpkTracker::default(),
            descriptors: BTreeMap::default(),
        }
    }
}

impl<K> Deref for KeychainTracker<K> {
    type Target = SpkTracker<(K, u32)>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K> DerefMut for KeychainTracker<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<K: Clone + Ord> KeychainTracker<K> {
    pub fn iter_descriptors(
        &self,
        range: impl core::ops::RangeBounds<K>,
    ) -> impl DoubleEndedIterator<Item = (K, &Descriptor<DescriptorPublicKey>)> {
        self.descriptors
            .range(range)
            .map(|(keychain, descriptor)| (keychain.clone(), descriptor))
    }

    pub fn add_keychain(&mut self, keychain: K, descriptor: Descriptor<DescriptorPublicKey>) {
        // TODO: panic if already different descriptor at that keychain
        self.descriptors.insert(keychain, descriptor);
    }

    pub fn descriptor(&self, keychain: K) -> &Descriptor<DescriptorPublicKey> {
        self.descriptors
            .get(&keychain)
            .expect("keychain does not exist")
    }

    ///
    pub fn next_derivation_index(&self, keychain: K) -> u32 {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .last()
            .map(|((_, last), _)| last + 1)
            .unwrap_or(0)
    }
    /// Derives script pubkeys from the descriptor **up to and including** `end` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn derive_spks(&mut self, keychain: K, end: u32) -> bool {
        let descriptor = self
            .descriptors
            .get(&keychain)
            .expect("no descriptor for keychain");
        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => end,
        };
        let next_to_derive = self.next_derivation_index(keychain.clone());
        if next_to_derive > end {
            return false;
        }

        for index in next_to_derive..=end {
            let spk = descriptor
                .at_derivation_index(index)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            self.inner.add_spk((keychain.clone(), index), spk);
        }

        true
    }

    /// Derives a new script pubkey for a keychain.
    ///
    /// The tracker returns a new script pubkey for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn derive_new(&mut self, keychain: K) -> (u32, &Script) {
        let secp = Secp256k1::verification_only();
        let next_derivation_index = self.next_derivation_index(keychain.clone());
        let descriptor = self
            .descriptors
            .get(&keychain)
            .expect("no descriptor for keychain");

        let new_spk = descriptor
            .at_derivation_index(next_derivation_index as u32)
            .derived_descriptor(&secp)
            .expect("the descriptor cannot need hardened derivation")
            .script_pubkey();

        let index = (keychain.clone(), next_derivation_index);
        self.inner.add_spk(index.clone(), new_spk);
        let new_spk = self
            .inner
            .script_pubkeys()
            .get(&index)
            .expect("we just added it");
        (next_derivation_index, new_spk)
    }

    pub fn derive_next_unused(&mut self, keychain: K) -> (u32, &Script) {
        let need_new = self.inner.iter_unused().next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(keychain)
        } else {
            self.inner
                .iter_unused()
                .filter(|((kc, _), _)| kc == &keychain)
                .map(|((_, i), script)| (i, script))
                .next()
                .unwrap()
        }
    }

    pub fn create_psbt(
        &self,
        inputs: impl IntoIterator<Item = OutPoint>,
        outputs: impl IntoIterator<Item = TxOut>,
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
            let definite_descriptor = self.get_txout(txin.previous_output).map(|ltxo| {
                self.descriptor(ltxo.spk_index.0)
                    .at_derivation_index(ltxo.spk_index.1)
            });

            if let Some(definite_descriptor) = definite_descriptor {
                let prev_tx = self
                    .get_tx(txin.previous_output.txid)
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

#[cfg(test)]
mod test {
    use crate::{
        ApplyResult, BlockId, BlockTime, CheckpointCandidate, KeychainTracker, PrevOuts,
        SpkTracker, Vec,
    };
    use bitcoin::{
        secp256k1::Secp256k1, BlockHash, OutPoint, Script, Transaction, TxIn, TxOut, Txid,
    };
    use miniscript::{Descriptor, DescriptorPublicKey};

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
                        match tx_spec.is_coinbase {
                            false => PrevOuts::Spend(
                                tx_spec
                                    .inputs
                                    .iter()
                                    .map(|in_spec| match in_spec {
                                        IOSpec::Mine(value, index) => TxOut {
                                            value: *value,
                                            script_pubkey: self
                                                .descriptor
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
        let mut tracker = KeychainTracker::default();
        tracker.add_keychain((), update_gen.descriptor.clone());
        tracker.derive_spks((), 0);

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
        let mut update_gen = UpdateGen::new();
        let mut tracker = KeychainTracker::default();
        tracker.add_keychain((), update_gen.descriptor.clone());
        tracker.derive_spks((), 0);

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
        let mut tracker = SpkTracker::<u32>::default();

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
        let mut tracker = SpkTracker::<u32>::default();

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
        let mut tracker = SpkTracker::<u32>::default();
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
        let mut tracker = SpkTracker::<u32>::default();
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
        let mut tracker = SpkTracker::<u32>::default();

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
