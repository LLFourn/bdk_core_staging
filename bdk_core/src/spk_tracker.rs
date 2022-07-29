use crate::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use crate::{FullTxOut, SparseChain, TxAtBlock};
use bitcoin::{self, hashes::sha256, OutPoint, Script, Txid};

/// A *script pubkey* tracker.
///
/// Given a list of script pubkeys a `SpkTracker` finds transaction outputs with those script
/// pubkeys. Keeps track of the application provided index (type paramter `I`) for each script Pubkey.
// The implementation of SpkTracker attempts to be stateless in that it doesn't need to know when
// there has been a re-org and transactions have disappeared or have been added. It keeps track of
// every output that it has ever seen but only returns those that are in the current sparse chain
// which is usually passed into the getter methods.
//
// To avoid rescanning every transaction when with sync with a sparse chain we keep track of a list
// of txid digests that we've seen and only apply the transactions between that digest and the new one.
//
// The expecption to the statelessness this is tracking of unspent txouts which relies on a concrete
// transaction graph. It is inefficient to iterate through every txout and check if it's unspent so
// we keep an uncompressed cache of all unspent outpoints at each txid digest we've seen.
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
    /// A set of previous states the SpkTracker has seen.
    /// The first tuple item in each element is the txid_digest and the second is the list of unspent txouts at that point.
    state_cache: VecDeque<(sha256::Hash, HashSet<OutPoint>)>,
    /// unconfirmed outpoints
    unconfirmed: HashSet<OutPoint>,
    unspent: HashSet<OutPoint>,
    cache_limit: usize,
}

impl<I> Default for SpkTracker<I> {
    fn default() -> Self {
        Self {
            txouts: Default::default(),
            unspent: Default::default(),
            script_pubkeys: Default::default(),
            spk_indexes: Default::default(),
            spk_txouts: Default::default(),
            unused: Default::default(),
            state_cache: Default::default(),
            unconfirmed: Default::default(),
            cache_limit: 100,
        }
    }
}

impl<I: Clone + Ord> SpkTracker<I> {
    pub fn sync(&mut self, chain: &SparseChain) {
        let point_of_agreement = self.state_cache.iter().find_map(|(txid_digest, unspent)| {
            Some((chain.checkpoint_with_txid_digest(*txid_digest)?, unspent))
        });

        let chain_checkpoint_range_to_apply = match point_of_agreement {
            Some((checkpoint_id, unspent_at_that_point)) => {
                self.unspent = unspent_at_that_point.clone();
                checkpoint_id.height + 1
            }
            None => {
                self.unspent.clear();
                0
            }
        };

        let txids_to_add = chain
            .iter_checkpoints(chain_checkpoint_range_to_apply..)
            .flat_map(|chain_checkpoint| chain.checkpoint_txids(chain_checkpoint))
            .chain(chain.unconfirmed().iter().cloned());

        for txid_to_add in txids_to_add {
            self.add_tx(txid_to_add, chain);
        }

        if let Some(latest_checkpoint) = chain.latest_checkpoint() {
            self.state_cache.push_front((
                chain.txid_digest_at(latest_checkpoint),
                self.unspent.clone(),
            ));
            self.state_cache.truncate(self.cache_limit);
        }
    }

    fn add_tx(&mut self, txid: Txid, chain: &SparseChain) {
        let TxAtBlock {
            tx,
            confirmation_time,
        } = chain.get_tx(txid).expect("must exist");
        for (i, out) in tx.output.iter().enumerate() {
            if let Some(index) = self.index_of_spk(&out.script_pubkey) {
                let outpoint = OutPoint {
                    txid,
                    vout: i as u32,
                };

                match confirmation_time {
                    Some(_) => {
                        self.unconfirmed.remove(&outpoint);
                    }
                    None => {
                        self.unconfirmed.insert(outpoint);
                    }
                }

                self.txouts.insert(outpoint, index.clone());

                let txos_for_script = self.spk_txouts.entry(index.clone()).or_default();
                txos_for_script.insert(outpoint);
                if chain.outspend(outpoint).is_none() {
                    self.unspent.insert(outpoint);
                }
                self.unused.remove(&index);
            }
        }

        for input in tx.input.iter() {
            self.unspent.remove(&input.previous_output);
        }
    }

    /// Iterate over unspent transactions outputs (i.e. UTXOs).
    pub fn iter_unspent(&self) -> impl Iterator<Item = (I, OutPoint)> + '_ {
        self.unspent.iter().map(|outpoint| {
            (
                self.txouts.get(outpoint).expect("must exist").clone(),
                *outpoint,
            )
        })
    }

    pub fn iter_unspent_full<'a>(
        &'a self,
        chain: &'a SparseChain,
    ) -> impl Iterator<Item = (I, FullTxOut)> + 'a {
        self.unspent.iter().filter_map(|outpoint| {
            debug_assert!(chain.full_txout(*outpoint).is_some());
            Some((
                self.txouts.get(outpoint).expect("must exist").clone(),
                chain.full_txout(*outpoint)?,
            ))
        })
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
    /// This returns `Some` if the txout has been found with a script pubkey in the tracker.
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
}
