use crate::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    FullTxOut, SparseChain, Vec,
};
use bitcoin::{self, hashes::sha256, OutPoint, Script, Transaction, Txid};

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
    script_pubkeys: BTreeMap<I, Script>,
    /// A reverse lookup from out script_pubkeys to derivation index
    spk_indexes: HashMap<Script, I>,
    /// A set of unused derivation indices.
    unused: BTreeSet<I>,
    /// Index the Outpoints owned by this tracker to the index of script pubkey.
    txouts: BTreeMap<OutPoint, I>,
    /// A lookup from script pubkey derivation index to related outpoints
    spk_txouts: BTreeMap<I, HashSet<OutPoint>>,
    /// A set of previous txid digests the tracker has seen.
    txid_digests_seen: HashSet<sha256::Hash>,
}

impl<I> Default for SpkTracker<I> {
    fn default() -> Self {
        Self {
            txouts: Default::default(),
            script_pubkeys: Default::default(),
            spk_indexes: Default::default(),
            spk_txouts: Default::default(),
            unused: Default::default(),
            txid_digests_seen: Default::default(),
        }
    }
}

impl<I: Clone + Ord> SpkTracker<I> {
    pub fn sync(&mut self, chain: &SparseChain) {
        // by reverse iterating the checkpoints and collecting the transactions until we find a
        // checkpoint with a txid digest we've seen we avoid having to apply *all* the transactions
        // in the sparse chain every time.
        let txids_to_add = chain
            .iter_checkpoints(..)
            .rev()
            .take_while(|checkpoint_id| {
                !self
                    .txid_digests_seen
                    .contains(&chain.txid_digest_at(*checkpoint_id))
            })
            .flat_map(|checkpoint_id| chain.checkpoint_txids(checkpoint_id))
            .chain(chain.unconfirmed().iter().cloned())
            .collect::<Vec<_>>();

        for txid_to_add in txids_to_add {
            self.add_tx(txid_to_add, chain);
        }

        if let Some(latest_checkpoint) = chain.latest_checkpoint() {
            let latest_txid_digest = chain.txid_digest_at(latest_checkpoint);
            self.txid_digests_seen.insert(latest_txid_digest);
        }
    }

    fn add_tx(&mut self, txid: Txid, chain: &SparseChain) {
        let tx = &chain.get_tx(txid).expect("must exist").tx;
        for (i, out) in tx.output.iter().enumerate() {
            if let Some(index) = self.index_of_spk(&out.script_pubkey) {
                let outpoint = OutPoint {
                    txid,
                    vout: i as u32,
                };

                self.txouts.insert(outpoint, index.clone());

                let txos_for_script = self.spk_txouts.entry(index.clone()).or_default();
                txos_for_script.insert(outpoint);
                self.unused.remove(&index);
            }
        }
    }

    /// Iterate over unspent transactions outputs (i.e. UTXOs).
    pub fn iter_unspent<'a>(
        &'a self,
        chain: &'a SparseChain,
    ) -> impl Iterator<Item = (I, OutPoint)> + '_ {
        // TODO: index unspent txouts somewhow
        self.iter_txout(chain)
            .filter(|(_, outpoint)| chain.outspend(*outpoint).is_none())
    }

    /// Convience method for retreiving  the same txouts [`iter_unspent`] gives and turning each outpoint into a `FullTxOut`
    /// using data from `chain`.
    ///
    /// [`iter_unspent`]: Self::iter_unspent
    pub fn iter_unspent_full<'a>(
        &'a self,
        chain: &'a SparseChain,
    ) -> impl Iterator<Item = (I, FullTxOut)> + 'a {
        self.iter_txout_full(chain)
            .filter(|(_, txout)| txout.spent_by.is_none())
    }

    /// Iterate over all the transaction outputs disovered by the tracker along with their
    /// associated script index.
    pub fn iter_txout_full<'a>(
        &'a self,
        chain: &'a SparseChain,
    ) -> impl DoubleEndedIterator<Item = (I, FullTxOut)> + 'a {
        self.txouts.iter().filter_map(|(outpoint, spk_index)| {
            Some((spk_index.clone(), chain.full_txout(*outpoint)?))
        })
    }

    pub fn iter_txout<'a>(
        &'a self,
        chain: &'a SparseChain,
    ) -> impl DoubleEndedIterator<Item = (I, OutPoint)> + 'a {
        self.txouts
            .iter()
            .filter(|(outpoint, _)| chain.get_tx(outpoint.txid).is_some())
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
    pub fn spk_at_index(&self, index: I) -> Option<&Script> {
        self.script_pubkeys.get(&index)
    }

    /// Iterate over the script pubkeys that have been derived already
    pub fn script_pubkeys(&self) -> &BTreeMap<I, Script> {
        &self.script_pubkeys
    }

    /// Adds a script pubkey to the tracker.
    ///
    /// The tracker will look for transactions spending to/from this script pubkey on all checkpoints
    /// that are subsequently added.
    pub fn add_spk(&mut self, index: I, spk: Script) {
        self.spk_indexes.insert(spk.clone(), index.clone());
        self.script_pubkeys.insert(index.clone(), spk);
        self.unused.insert(index);
    }

    /// Iterate over the script pubkeys that have been derived but do not have a transaction spending to them.
    pub fn iter_unused(&self) -> impl Iterator<Item = (I, &Script)> {
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
