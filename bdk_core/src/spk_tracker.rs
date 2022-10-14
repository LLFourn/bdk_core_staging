use crate::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    FullTxOut, SparseChain, TxGraph,
};
use bitcoin::{self, OutPoint, Script, Transaction};

/// A *script pubkey* tracker.
///
/// Given a list of script pubkeys a `SpkTracker` finds transaction outputs with those script
/// pubkeys. Keeps track of the application provided index (type paramter `I`) for each script Pubkey.
// The implementation of SpkTracker attempts to be *monotone* in that it never deletes transactions
// from its internal index. It keeps track of every output that it has ever seen. It only returns
// those that are in the current sparse chain which is usually passed into the getter methods.
//
// To avoid rescanning every transaction when with sync with a sparse chain we keep a set of txid
// digests that we've seen and only apply the transactions between where we found that digest and
// the tip of the chain.
#[derive(Clone, Debug)]
pub struct SpkTracker<I> {
    /// Derived script_pubkeys ordered by derivation index.
    script_pubkeys: BTreeMap<I, (Script, HashSet<OutPoint>)>,
    /// A reverse lookup from out script_pubkeys to derivation index
    spk_indexes: HashMap<Script, I>,
    /// A set of unused derivation indices.
    unused: BTreeSet<I>,
    /// Index the Outpoints owned by this tracker to the index of script pubkey.
    txouts: BTreeMap<OutPoint, I>,
}

impl<I> Default for SpkTracker<I> {
    fn default() -> Self {
        Self {
            txouts: Default::default(),
            script_pubkeys: Default::default(),
            spk_indexes: Default::default(),
            unused: Default::default(),
        }
    }
}

impl<I: Clone + Ord> SpkTracker<I> {
    pub fn sync(&mut self, chain: &SparseChain, graph: &TxGraph) {
        chain
            .iter_txids()
            .map(|(_, txid)| graph.tx(&txid).expect("tx must exist"))
            .for_each(|tx| self.add_tx(tx));
    }

    fn add_tx(&mut self, tx: &Transaction) {
        for (i, out) in tx.output.iter().enumerate() {
            if let Some(index) = self.spk_indexes.get(&out.script_pubkey) {
                let outpoint = OutPoint {
                    txid: tx.txid(),
                    vout: i as u32,
                };

                let (_, ops) = self.script_pubkeys.get_mut(index).expect("should exist");
                ops.insert(outpoint);

                self.txouts.insert(outpoint, index.clone());
                self.unused.remove(&index);
            }
        }
    }

    /// Iterate over unspent transactions outputs (i.e. UTXOs).
    pub fn iter_unspent<'a>(
        &'a self,
        chain: &'a SparseChain,
        graph: &'a TxGraph,
    ) -> impl Iterator<Item = (I, OutPoint)> + '_ {
        // TODO: index unspent txouts somewhow
        self.iter_txout(graph)
            .filter(|(_, outpoint)| chain.transaction_at(&outpoint.txid).is_some())
            .filter(|(_, outpoint)| graph.is_unspent(outpoint).expect("should exist"))
    }

    /// Convience method for retreiving  the same txouts [`iter_unspent`] gives and turning each outpoint into a `FullTxOut`
    /// using data from `chain`.
    ///
    /// [`iter_unspent`]: Self::iter_unspent
    pub fn iter_unspent_full<'a>(
        &'a self,
        chain: &'a SparseChain,
        graph: &'a TxGraph,
    ) -> impl Iterator<Item = (I, FullTxOut)> + 'a {
        self.iter_txout_full(chain, graph)
            .filter(|(_, txout)| txout.spent_by.is_none())
    }

    /// Iterate over all the transaction outputs disovered by the tracker along with their
    /// associated script index.
    pub fn iter_txout_full<'a>(
        &'a self,
        chain: &'a SparseChain,
        graph: &'a TxGraph,
    ) -> impl DoubleEndedIterator<Item = (I, FullTxOut)> + 'a {
        self.txouts.iter().filter_map(|(outpoint, spk_index)| {
            Some((spk_index.clone(), chain.full_txout(graph, *outpoint)?))
        })
    }

    pub fn iter_txout<'a>(
        &'a self,
        graph: &'a TxGraph,
    ) -> impl DoubleEndedIterator<Item = (I, OutPoint)> + 'a {
        self.txouts
            .iter()
            .filter(|(outpoint, _)| graph.tx(&outpoint.txid).is_some())
            .map(|(op, index)| (index.clone(), *op))
    }

    /// Returns the index of the script pubkey at `outpoint`.
    ///
    /// This returns `Some(spk_index)` if the txout has been found with a script pubkey in the tracker.
    pub fn index_of_txout(&self, outpoint: OutPoint) -> Option<I> {
        self.txouts.get(&outpoint).cloned()
    }

    /// Returns the script that has been derived at the index.
    ///
    /// If that index hasn't been derived yet it will return `None`.
    pub fn spk_at_index(&self, index: I) -> Option<&(Script, HashSet<OutPoint>)> {
        self.script_pubkeys.get(&index)
    }

    /// Iterate over the script pubkeys that have been derived already
    pub fn script_pubkeys(&self) -> &BTreeMap<I, (Script, HashSet<OutPoint>)> {
        &self.script_pubkeys
    }

    /// Adds a script pubkey to the tracker.
    ///
    /// The tracker will look for transactions spending to/from this script pubkey on all checkpoints
    /// that are subsequently added.
    pub fn add_spk(&mut self, index: I, spk: Script) -> Result<bool, Script> {
        let (entry_spk, entry_ops) = self
            .script_pubkeys
            .entry(index.clone())
            .or_insert_with(|| (spk.clone(), HashSet::new()));

        if entry_spk != &spk {
            return Err(entry_spk.clone());
        }

        self.spk_indexes
            .entry(spk.clone())
            .or_insert_with(|| index.clone());

        let changed = if entry_ops.is_empty() {
            self.unused.insert(index)
        } else {
            false
        };

        Ok(changed)
    }

    /// Iterate over the script pubkeys that have been derived but do not have a transaction spending to them.
    pub fn iter_unused(&self) -> impl Iterator<Item = (I, &Script)> {
        self.unused.iter().map(|index| {
            (
                index.clone(),
                &self.spk_at_index(index.clone()).expect("must exist").0,
            )
        })
    }

    /// Returns whether the script pubkey at index `index` has been used or not.
    ///
    /// i.e. has a transaction which spends to it.
    pub fn is_used(&self, index: I) -> bool {
        self.script_pubkeys
            .get(&index)
            .map(|(_, ops)| !ops.is_empty())
            .unwrap_or(false)
    }

    /// Returns the index associated with the script pubkey.
    pub fn index_of_spk(&self, script: &Script) -> Option<I> {
        self.spk_indexes.get(script).cloned()
    }

    /// Whether any of the inputs of this transaction spend a txout tracked or whether any output
    /// matches one of our script pubkeys.
    pub fn is_relevant(&self, tx: &Transaction) -> bool {
        let input_matches = tx
            .input
            .iter()
            .find(|input| self.index_of_txout(input.previous_output).is_some())
            .is_some();
        let output_matches = tx
            .output
            .iter()
            .find(|output| self.index_of_spk(&output.script_pubkey).is_some())
            .is_some();
        input_matches || output_matches
    }
}
