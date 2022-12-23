use bitcoin::{Transaction, Txid};
use miniscript::{Descriptor, DescriptorPublicKey};

use crate::{
    chain_graph::{self, ChainGraph},
    collections::*,
    keychain::{KeychainChangeSet, KeychainScan},
    keychain_txout_index::KeychainTxOutIndex,
    sparse_chain::{self, SparseChain},
    tx_graph::TxGraph,
    BlockId, FullTxOut,
};

/// A convenient combination of a `KeychainTxOutIndex<K>` and a `ChainGraph<I>`.
///
/// The `KeychainTracker<K, I>` atomically updates its `KeychainTxOutIndex<K>` whenever new chain data is
/// incorporated into its internal `chain_graph`.
///
/// [`KeychainTxOutIndex<K>`]: crate::KeychainTxOutIndex
#[derive(Clone, Debug)]
pub struct KeychainTracker<K, I> {
    /// Index between script pubkeys to transaction outputs
    pub txout_index: KeychainTxOutIndex<K>,
    chain_graph: ChainGraph<I>,
}

impl<K, I> KeychainTracker<K, I>
where
    I: sparse_chain::ChainIndex,
    K: Ord + Clone + core::fmt::Debug,
{
    /// Add a keychain to the tracker's `txout_index` with a descriptor to derive addresses for it.
    /// This is just shorthand for calling [`KeychainTxOutIndex::add_keychain`] on the internal
    /// `txout_index`.
    ///
    /// Adding a keychain means you will be able to derive new script pubkeys under that keychain
    /// and the tracker will discover transaction outputs with those script pubkeys.
    pub fn add_keychain(&mut self, keychain: K, descriptor: Descriptor<DescriptorPublicKey>) {
        self.txout_index.add_keychain(keychain, descriptor)
    }

    /// Get the internal map of keychains to their descriptors. This is just shorthand for calling
    /// [`KeychainTxOutIndex::keychains`] on the internal `txout_index`.
    pub fn keychains(&mut self) -> &BTreeMap<K, Descriptor<DescriptorPublicKey>> {
        self.txout_index.keychains()
    }

    pub fn checkpoint_limit(&self) -> Option<usize> {
        self.chain_graph.checkpoint_limit()
    }

    pub fn set_checkpoint_limit(&mut self, limit: Option<usize>) {
        self.chain_graph.set_checkpoint_limit(limit)
    }

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

    pub fn apply_changeset(
        &mut self,
        changeset: KeychainChangeSet<K, I>,
    ) -> Result<(), (KeychainChangeSet<K, I>, HashSet<Txid>)> {
        self.txout_index
            .store_all_up_to(&changeset.derivation_indices);
        self.txout_index.scan(&changeset);
        let derivation_indices = changeset.derivation_indices;
        self.chain_graph
            .apply_changeset(changeset.chain_graph)
            .map_err(|(cg_changeset, missing)| {
                (
                    KeychainChangeSet {
                        derivation_indices,
                        chain_graph: cg_changeset,
                    },
                    missing,
                )
            })
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

    pub fn chain_graph(&self) -> &ChainGraph<I> {
        &self.chain_graph
    }

    pub fn graph(&self) -> &TxGraph {
        &self.chain_graph().graph()
    }

    pub fn chain(&self) -> &SparseChain<I> {
        &self.chain_graph().chain()
    }

    /// Insert a `block_id` (a height and block hash) into the chain. The caller is responsible for
    /// guaranteeing that a block exists at that height. If a checkpoint already exists at that
    /// height with a different hash this will return an error. Otherwise it will return `Ok(true)`
    /// if the checkpoint didn't already exist or `Ok(false)` if it did.
    ///
    /// **Warning**: This function modifies the internal state of the tracker. You are responsible
    /// for persisting these changes to disk if you need to restore them.
    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<bool, sparse_chain::InsertCheckpointErr> {
        self.chain_graph.insert_checkpoint(block_id)
    }

    /// Inserts a transaction into the inner [`ChainGraph`] and optionally into the inner chain at
    /// `position`.
    ///
    /// **Warning**: This function modifies the internal state of the chain graph. You are
    /// responsible for persisting these changes to disk if you need to restore them.
    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        position: Option<I>,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        let changed = self.chain_graph.insert_tx(tx.clone(), position)?;
        self.txout_index.scan(&tx);
        Ok(changed)
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
