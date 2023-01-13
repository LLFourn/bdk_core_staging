use bitcoin::Transaction;
use miniscript::{Descriptor, DescriptorPublicKey};

use crate::{
    chain_graph::{self, ChainGraph},
    collections::*,
    keychain::{KeychainChangeSet, KeychainScan, KeychainTxOutIndex},
    sparse_chain::{self, SparseChain},
    tx_graph::TxGraph,
    BlockId, FullTxOut, TxHeight,
};

use super::Balance;

/// A convenient combination of a `KeychainTxOutIndex<K>` and a `ChainGraph<P>`.
///
/// The `KeychainTracker<K, P>` atomically updates its `KeychainTxOutIndex<K>` whenever new chain data is
/// incorporated into its internal `chain_graph`.
///
/// [`KeychainTxOutIndex<K>`]: crate::KeychainTxOutIndex
#[derive(Clone, Debug)]
pub struct KeychainTracker<K, P> {
    /// Index between script pubkeys to transaction outputs
    pub txout_index: KeychainTxOutIndex<K>,
    chain_graph: ChainGraph<P>,
}

impl<K, P> KeychainTracker<K, P>
where
    P: sparse_chain::ChainPosition,
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
        scan: &KeychainScan<K, P>,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::UpdateError<P>> {
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

    pub fn apply_update(
        &mut self,
        scan: KeychainScan<K, P>,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::UpdateError<P>> {
        let changeset = self.determine_changeset(&scan)?;
        self.apply_changeset(changeset.clone())
            .expect("generated changeset should apply");
        Ok(changeset)
    }

    pub fn apply_changeset(
        &mut self,
        changeset: KeychainChangeSet<K, P>,
    ) -> Result<(), chain_graph::InflateError<P>> {
        self.txout_index
            .store_all_up_to(&changeset.derivation_indices);
        self.txout_index.scan(&changeset);
        self.chain_graph.apply_changeset(changeset.chain_graph)
    }

    pub fn full_txouts(&self) -> impl Iterator<Item = (&(K, u32), FullTxOut<P>)> + '_ {
        self.txout_index
            .txouts()
            .filter_map(|(spk_i, op, _)| Some((spk_i, self.chain_graph.full_txout(op)?)))
    }

    pub fn full_utxos(&self) -> impl Iterator<Item = (&(K, u32), FullTxOut<P>)> + '_ {
        self.full_txouts()
            .filter(|(_, txout)| txout.spent_by.is_none())
    }

    pub fn chain_graph(&self) -> &ChainGraph<P> {
        &self.chain_graph
    }

    pub fn graph(&self) -> &TxGraph {
        &self.chain_graph().graph()
    }

    pub fn chain(&self) -> &SparseChain<P> {
        &self.chain_graph().chain()
    }

    /// Insert a `block_id` (a height and block hash) into the chain. The caller is responsible for
    /// guaranteeing that a block exists at that height. If a checkpoint already exists at that
    /// height with a different hash this will return an error. Otherwise it will return `Ok(true)`
    /// if the checkpoint didn't already exist or `Ok(false)` if it did.
    ///
    /// **Warning**: This function modifies the internal state of the tracker. You are responsible
    /// for persisting these changes to disk if you need to restore them.
    pub fn insert_checkpoint_preview(
        &self,
        block_id: BlockId,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::InsertCheckpointError> {
        Ok(KeychainChangeSet {
            chain_graph: self.chain_graph.insert_checkpoint_preview(block_id)?,
            ..Default::default()
        })
    }

    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::InsertCheckpointError> {
        let changeset = self.insert_checkpoint_preview(block_id)?;
        self.apply_changeset(changeset.clone())
            .expect("changeset should apply");
        Ok(changeset)
    }

    /// Inserts a transaction into the inner [`ChainGraph`] and optionally into the inner chain at
    /// `position`.
    ///
    /// **Warning**: This function modifies the internal state of the chain graph. You are
    /// responsible for persisting these changes to disk if you need to restore them.
    pub fn insert_tx_preview(
        &self,
        tx: Transaction,
        pos: P,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::InsertTxError<P>> {
        Ok(KeychainChangeSet {
            chain_graph: self.chain_graph.insert_tx_preview(tx.clone(), pos)?,
            ..Default::default()
        })
    }

    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        pos: P,
    ) -> Result<KeychainChangeSet<K, P>, chain_graph::InsertTxError<P>> {
        let changeset = self.insert_tx_preview(tx, pos)?;
        self.apply_changeset(changeset.clone())
            .expect("changeset should apply");
        Ok(changeset)
    }

    /// Returns the *balance* of the keychain i.e. the value of unspent transaction outputs tracked.
    /// The caller provides a `should_trust` predicate which must decide whether the value of
    /// unconfirmed outputs on this keychain are guaranteed to be realized or not. For example:
    ///
    /// - For an *internal* (change) keychain `should_trust` should in general be `true` since even if
    /// you lose an internal output due to eviction you will always gain back the value from whatever output the
    /// unconfirmed transaction was spending (since that output is presumeably from your wallet).
    /// - For an *external* keychain you might want `should_trust` to return  `false` since someone may cancel (by double spending)
    /// a payment made to addresses on that keychain.
    ///
    /// When in doubt set `should_trust` to return false. This doesn't do anything other than change
    /// where the unconfirmed output's value is accounted for in `Balance`.
    pub fn balance(&self, mut should_trust: impl FnMut(&K) -> bool) -> Balance {
        let mut immature = 0;
        let mut trusted_pending = 0;
        let mut untrusted_pending = 0;
        let mut confirmed = 0;
        let last_sync_height = self.chain().latest_checkpoint().map(|latest| latest.height);
        for ((keychain, _), utxo) in self.full_utxos() {
            let chain_position = &utxo.chain_position;

            match chain_position.height() {
                TxHeight::Confirmed(_) => {
                    if utxo.is_on_coinbase {
                        if utxo.is_mature(
                            last_sync_height
                                .expect("since it's confirmed we must have a checkpoint"),
                        ) {
                            confirmed += utxo.txout.value;
                        } else {
                            immature += utxo.txout.value;
                        }
                    } else {
                        confirmed += utxo.txout.value;
                    }
                }
                TxHeight::Unconfirmed => {
                    if should_trust(keychain) {
                        trusted_pending += utxo.txout.value;
                    } else {
                        untrusted_pending += utxo.txout.value;
                    }
                }
            }
        }

        Balance {
            immature,
            trusted_pending,
            untrusted_pending,
            confirmed,
        }
    }

    /// Returns the balance of all spendable confirmed unspent outputs of this tracker at a
    /// particular height.
    pub fn balance_at(&self, height: u32) -> u64 {
        self.full_txouts()
            .filter(|(_, full_txout)| full_txout.is_spendable_at(height))
            .map(|(_, full_txout)| full_txout.txout.value)
            .sum()
    }
}

impl<K, P> Default for KeychainTracker<K, P> {
    fn default() -> Self {
        Self {
            txout_index: Default::default(),
            chain_graph: Default::default(),
        }
    }
}

impl<K, P> AsRef<TxGraph> for KeychainTracker<K, P> {
    fn as_ref(&self) -> &TxGraph {
        self.chain_graph.as_ref()
    }
}
