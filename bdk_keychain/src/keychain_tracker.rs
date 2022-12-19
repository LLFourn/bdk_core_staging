use bdk_core::{
    chain_graph::{self, ChainGraph},
    keychain::{KeychainChangeSet, KeychainScan},
    sparse_chain::{self, SparseChain},
    tx_graph::TxGraph,
    FullTxOut,
};
use miniscript::plan::{Assets, CanDerive, Plan};

use crate::KeychainTxOutIndex;

/// A convenient combination of a `KeychainTxOutIndex<K>` and a `ChainGraph<I>`.
///
/// The `KeychainTracker<K, I>` atomically updates its `KeychainTxOutIndex<K>` whenever new chain data is
/// incorporated into its internal `chain_graph`.
///
/// [`KeychainTxOutIndex<K>`]: crate::KeychainTxOutIndex
#[derive(Clone, Debug)]
pub struct KeychainTracker<K, I> {
    /// script pubkey index
    pub txout_index: KeychainTxOutIndex<K>,
    chain_graph: ChainGraph<I>,
}

impl<K, I> KeychainTracker<K, I>
where
    I: sparse_chain::ChainIndex,
    K: Ord + Clone + core::fmt::Debug,
{
    pub fn determine_changeset(
        &self,
        scan: &KeychainScan<K, I>,
    ) -> Result<KeychainChangeSet<K, I>, chain_graph::UpdateFailure<I>> {
        let mut new_derivation_indices = scan.last_active_indexes.clone();
        new_derivation_indices.retain(|keychain, index| {
            match self.txout_index.derivation_index(keychain) {
                Some(existing) => *index > existing,
                None => true,
            }
        });

        Ok(KeychainChangeSet {
            derivation_indices: new_derivation_indices,
            chain_graph: self.chain_graph.determine_changeset(&scan.update)?,
        })
    }

    pub fn apply_changeset(&mut self, changeset: KeychainChangeSet<K, I>) {
        self.txout_index
            .store_all_up_to(&changeset.derivation_indices);
        self.txout_index.scan(&changeset);
        self.chain_graph.apply_changeset(changeset.chain_graph);
    }

    pub fn full_txouts(&self) -> impl Iterator<Item = (&(K, u32), FullTxOut<I>)> + '_ {
        self.txout_index
            .txouts()
            .filter_map(|(spk_i, op, _)| Some((spk_i, self.chain_graph.full_txout(op)?)))
    }

    pub fn full_utxos(&self) -> impl Iterator<Item = (&(K, u32), FullTxOut<I>)> + '_ {
        self.full_txouts()
            .filter(|(_, txout)| txout.spent_by.is_none())
    }

    pub fn planned_utxos<'a, AK: CanDerive + Clone>(
        &'a self,
        assets: &'a Assets<AK>,
    ) -> impl Iterator<Item = (Plan<AK>, FullTxOut<I>)> + 'a {
        self.full_utxos()
            .filter_map(|((keychain, derivation_index), full_txout)| {
                Some((
                    self.txout_index
                        .keychains()
                        .get(keychain)
                        .expect("must exist since we have a utxo for it")
                        .at_derivation_index(*derivation_index)
                        .plan_satisfaction(assets)?,
                    full_txout,
                ))
            })
    }

    pub fn chain_graph(&self) -> &ChainGraph<I> {
        &self.chain_graph
    }

    pub fn graph(&self) -> &TxGraph {
        &self.chain_graph().graph()
    }

    pub fn chain(&self) -> &SparseChain<I> {
        &self.chain_graph().chain()
    }
}

impl<K, I> Default for KeychainTracker<K, I> {
    fn default() -> Self {
        Self {
            txout_index: Default::default(),
            chain_graph: Default::default(),
        }
    }
}

impl<K, I> AsRef<TxGraph> for KeychainTracker<K, I> {
    fn as_ref(&self) -> &TxGraph {
        self.chain_graph.as_ref()
    }
}
