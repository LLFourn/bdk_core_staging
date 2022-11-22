use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use bitcoin::{self, OutPoint, Script, Transaction, TxOut, Txid};

/// An index storing [`TxOut`]s that had a script pubkey that matches those in an updatable list.
/// The basic idea is that you insert script pubkeys you care about into the index with [`add_spk`]
/// and then the index will look at any transaction data you pass into and store any `TxOut`s with
/// those script pubkeys.
///
/// Each script pubkey is associated with a application defined index script index `I` which must be
/// [`Ord`]. Usually this is used to store the derivation index of the script pubkey or even a
/// combination of `(keychain, derivation_index)`.
///
/// Note that `SpkTxOutIndex` is intentionally *monotone* -- you cannot delete or modify txouts that
/// it has indexed. It doesn't care about confirmation height of the txouts it indexes. To track
/// that information use a [`SparseChain`].
///
/// [`TxOut`]: bitcoin::TxOut
/// [`add_spk`]: Self::add_spk
/// [`Ord`]: core::cmp::Ord
/// [`SparseChain`]: crate::sparse_chain::SparseChain
#[derive(Clone, Debug)]
pub struct SpkTxOutIndex<I> {
    /// Derived script_pubkeys ordered by derivation index.
    script_pubkeys: BTreeMap<I, Script>,
    /// A reverse lookup from out script_pubkeys to derivation index
    spk_indexes: HashMap<Script, I>,
    /// A set of unused derivation indices.
    unused: BTreeSet<I>,
    /// Index the `OutPoint` to the index of script pubkey.
    txouts: BTreeMap<OutPoint, (I, TxOut)>,
    /// A lookup from script pubkey derivation index to related outpoints
    spk_txouts: BTreeMap<I, HashSet<OutPoint>>,
}

impl<I> Default for SpkTxOutIndex<I> {
    fn default() -> Self {
        Self {
            txouts: Default::default(),
            script_pubkeys: Default::default(),
            spk_indexes: Default::default(),
            spk_txouts: Default::default(),
            unused: Default::default(),
        }
    }
}

impl<I: Clone + Ord> SpkTxOutIndex<I> {
    /// Scans an objecting containing many txouts.
    ///
    /// Typically this is used in two situations:
    ///
    /// 1. After loading transaction data from disk you may scan over all the txouts to restore all
    /// your txouts.
    /// 2. When getting new data from the chain it is efficient to scan it first in isolation.
    ///
    /// See [`ForEachTxout`] for the types that support this.
    ///
    /// [`ForEachTxout`]: crate::ForEachTxout
    pub fn scan(&mut self, txouts: &impl ForEachTxout) {
        txouts.for_each_txout(&mut |(op, txout)| self.scan_txout(op, txout))
    }

    /// Scan a single `TxOut` for a matching script pubkey
    pub fn scan_txout(&mut self, op: OutPoint, txout: &TxOut) {
        if let Some(spk_i) = self.index_of_spk(&txout.script_pubkey) {
            self.txouts
                .insert(op.clone(), (spk_i.clone(), txout.clone()));
            self.spk_txouts
                .entry(spk_i.clone())
                .or_default()
                .insert(op.clone());
            self.unused.remove(&spk_i);
        }
    }

    /// Iterate over all known txouts that spend to tracked scriptPubKeys.
    pub fn iter_txout(
        &self,
    ) -> impl DoubleEndedIterator<Item = (I, OutPoint, &TxOut)> + ExactSizeIterator {
        self.txouts
            .iter()
            .map(|(op, (index, txout))| (index.clone(), *op, txout))
    }

    /// Finds all txouts on a transaction that has previously been scanned and indexed.
    pub fn txouts_in_tx(
        &self,
        txid: Txid,
    ) -> impl DoubleEndedIterator<Item = (I, OutPoint, &TxOut)> {
        self.txouts
            .range(OutPoint::new(txid, u32::MIN)..=OutPoint::new(txid, u32::MAX))
            .map(|(op, (index, txout))| (index.clone(), *op, txout))
    }

    /// Returns the txout and script pubkey index of the `TxOut` at `OutPoint`.
    ///
    /// Returns `None` if the `TxOut` hasn't been scanned or if nothing matching was found there.
    pub fn txout(&self, outpoint: OutPoint) -> Option<(I, &TxOut)> {
        self.txouts
            .get(&outpoint)
            .map(|(spk_i, txout)| (spk_i.clone(), txout))
    }

    /// Returns the script that has been inserted at the `index`.
    ///
    /// If that index hasn't been inserted yet it will return `None`.
    pub fn spk_at_index(&self, index: I) -> Option<&Script> {
        self.script_pubkeys.get(&index)
    }

    /// The script pubkeys being tracked by the index.
    pub fn script_pubkeys(&self) -> &BTreeMap<I, Script> {
        &self.script_pubkeys
    }

    /// Adds a script pubkey to scan for.
    ///
    /// the index will look for outputs spending to whenever it scans new data.
    pub fn add_spk(&mut self, index: I, spk: Script) {
        self.spk_indexes.insert(spk.clone(), index.clone());
        self.script_pubkeys.insert(index.clone(), spk);
        self.unused.insert(index);
    }

    /// Iterate over the script pubkeys that have been derived but do not have a transaction spending to them.
    pub fn iter_unused(&self) -> impl DoubleEndedIterator<Item = (I, &Script)> + ExactSizeIterator {
        self.unused.iter().map(|index| {
            (
                index.clone(),
                self.spk_at_index(index.clone()).expect("must exist"),
            )
        })
    }

    /// Returns whether the script pubkey at index `index` has been used or not.
    ///
    /// i.e. has a transaction which spends to it.
    pub fn is_used(&self, index: I) -> bool {
        self.spk_txouts
            .get(&index)
            .map(|set| !set.is_empty())
            .unwrap_or(false)
    }

    /// Returns the index associated with the script pubkey.
    pub fn index_of_spk(&self, script: &Script) -> Option<I> {
        self.spk_indexes.get(script).cloned()
    }

    /// Whether any of the inputs of this transaction spend a txout tracked or whether any output
    /// matches one of our script pubkeys.
    ///
    /// It is easily possible to misuse this method and get false negatives by calling it before you
    /// have scanned the `TxOut`s the transaction is spending. For example if you want to filter out
    /// all the transactions in a block that are irrelevant you **must first scan all the
    /// transactions in the block** and only then use this method.
    pub fn is_relevant(&self, tx: &Transaction) -> bool {
        let input_matches = tx
            .input
            .iter()
            .find(|input| self.txouts.contains_key(&input.previous_output))
            .is_some();
        let output_matches = tx
            .output
            .iter()
            .find(|output| self.spk_indexes.contains_key(&output.script_pubkey))
            .is_some();
        input_matches || output_matches
    }
}

/// Trait to do something with every txout contained in a structure. We'd prefer to use an iterator
/// here but rust's type system makes it extremely hard to do this (without trait objects).
pub trait ForEachTxout {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut)));
}

impl ForEachTxout for Transaction {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        let txid = self.txid();
        for (i, txout) in self.output.iter().enumerate() {
            f((
                OutPoint {
                    txid,
                    vout: i as u32,
                },
                txout,
            ))
        }
    }
}

impl ForEachTxout for bitcoin::Block {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        for tx in self.txdata.iter() {
            tx.for_each_txout(f)
        }
    }
}
