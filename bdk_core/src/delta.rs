use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    marker::PhantomData,
};

use bitcoin::{OutPoint, Transaction, Txid};

use super::*;

/// State of [Delta].
pub trait DeltaState {}
impl DeltaState for Unfilled {}
impl DeltaState for Filled {}
impl DeltaState for Negated {}

/// [Delta] is in an unfilled state (not all referenced transactions exist).
#[derive(Debug, Default, Clone, Copy)]
pub struct Unfilled;

/// [Delta] is in a filled state (we can apply to [SparseChain]).
#[derive(Debug, Default, Clone, Copy)]
pub struct Filled;

/// [Delta] is in a negated state (to remove data).
#[derive(Debug, Default, Clone, Copy)]
pub struct Negated;

/// Candidate changes of [SparseChain]
#[derive(Debug, Default, Clone)]
pub struct Delta<S: DeltaState> {
    pub(crate) blocks: BTreeMap<u32, PartialHeader>,
    pub(crate) tx_keys: BTreeSet<(u32, Txid)>,
    pub(crate) tx_values: HashMap<Txid, Transaction>, // needs to be filled for the `Filled` state

    pub(crate) marker: PhantomData<S>,
}

impl<S: DeltaState> Delta<S> {
    /// Returns `true` if the [Delta] is empty (no changes).
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty() && self.tx_keys.is_empty()
    }

    /// Iterates through transactions that are contained in [Delta].
    ///
    /// [Txid]s in which the raw transaction is missing, are skipped.
    pub fn iter_txs(&self) -> impl Iterator<Item = ((u32, Txid), Transaction)> + '_ {
        self.tx_keys
            .iter()
            .filter_map(move |k| self.tx_values.get(&k.1).map(|tx| (*k, tx.clone())))
    }

    pub fn iter_txouts(&self) -> impl Iterator<Item = (Txid, u32, TxOut)> + '_ {
        self.tx_keys
            .iter()
            .filter_map(move |k| self.tx_values.get(&k.1))
            .flat_map(|tx| {
                tx.output
                    .iter()
                    .enumerate()
                    .map(|(vout, txout)| (tx.txid(), vout as u32, txout.clone()))
            })
    }
}

impl Delta<Unfilled> {
    /// Iterates through missing txids.
    pub fn missing_txids(&self) -> impl Iterator<Item = Txid> + '_ {
        self.tx_keys
            .iter()
            .map(|(_, txid)| *txid)
            .filter(move |txid| !self.tx_values.contains_key(txid))
    }

    /// Fill all transactions.
    pub fn fill_transactions<I>(mut self, tx_iter: I) -> Result<Delta<Filled>, Self>
    where
        I: Iterator<Item = Transaction>,
    {
        self.tx_values.extend(tx_iter.map(|tx| (tx.txid(), tx)));

        if self.missing_txids().count() == 0 {
            Ok(Delta::<Filled> {
                blocks: self.blocks,
                tx_keys: self.tx_keys,
                tx_values: self.tx_values,
                marker: PhantomData,
            })
        } else {
            Err(self)
        }
    }
}

impl Delta<Filled> {
    /// Applies deltas to the given [SparseChain].
    ///
    // TODO: We can return an `AppliedToSparseChain` struct that records confirmed txs.
    pub fn apply_to_sparsechain(
        self,
        sparsechain: &mut AlternativeSparseChain,
    ) -> Result<(), CoreError> {
        let tx_heights = self
            .tx_keys
            .iter()
            .map(|(height, txid)| (*txid, *height))
            .collect::<Vec<_>>();

        let spends = self
            .tx_values
            .iter()
            .flat_map(|(txid, tx)| {
                tx.input
                    .iter()
                    .map(move |txin| (txin.previous_output, *txid))
                // .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let persistence_state = self.tx_keys.iter().cloned().next();

        // mark txs as confirmed: remove txs of `(u32::MAX, txid)` as unconfired txs are stored
        // with height u32::MAX
        //
        // TODO: We can return an `AppliedToSparseChain` struct that records confirmed txs.
        self.tx_keys.iter().for_each(|(_, txid)| {
            sparsechain.txs.remove(&(u32::MAX, *txid));
        });

        // update sparse chain
        sparsechain.blocks.extend(&self.blocks);
        sparsechain.txs.extend(self.iter_txs());
        sparsechain.at_height.extend(tx_heights);
        sparsechain.spends.extend(spends);
        sparsechain.persist_from = match (sparsechain.persist_from, persistence_state) {
            (None, Some(s)) => Some(s),
            (Some(s), None) => Some(s),
            (None, None) => None,
            (a, b) => std::cmp::min(a, b),
        };

        Ok(())
    }

    /// Applies the given delta to a [SpkTracker].
    pub fn apply_to_spk_tracker<I: Clone + Ord>(
        self,
        spk_tracker: &mut SpkTracker<AlternativeSparseChain, I>,
    ) {
        self.iter_txouts().for_each(|(txid, vout, txout)| {
            if let Some(index) = spk_tracker.index_of_spk(&txout.script_pubkey) {
                let outpoint = OutPoint { txid, vout };
                spk_tracker.txouts.insert(outpoint, index.clone());
                spk_tracker
                    .spk_txouts
                    .entry(index.clone())
                    .or_default()
                    .insert(outpoint);
                spk_tracker.unused.remove(&index);
            }
        })
    }
}

// TODO: This is not needed. `SpkTracker` is monotonic (it only grows).
impl Delta<Negated> {
    /// Applies the given negated delta to [SpkTracker].
    pub fn apply_to_spk_tracker<I: Clone + Ord>(
        self,
        spk_tracker: &mut SpkTracker<AlternativeSparseChain, I>,
    ) {
        self.iter_txouts().for_each(|(txid, vout, txout)| {
            if let Some(index) = spk_tracker.index_of_spk(&txout.script_pubkey) {
                let outpoint = OutPoint { txid, vout };

                if spk_tracker.txouts.remove(&outpoint).is_some() {
                    let empty = {
                        let spk_txouts = spk_tracker
                            .spk_txouts
                            .get_mut(&index)
                            .expect("inconsistent");
                        spk_txouts.remove(&outpoint);
                        spk_txouts.is_empty()
                    };

                    if empty {
                        spk_tracker.spk_txouts.remove(&index);
                    }
                }
            }
        })
    }
}
