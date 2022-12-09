use crate::{
    sparse_chain::{self, ChainIndex, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ForEachTxout, FullTxOut, TxHeight,
};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

/// A convenient combination of a [`SparseChain<I>`] and a [`TxGraph`].
///
/// Very often you want to store transaction data when you record a transaction's existence. Adding
/// a transaction to a `ChainGraph` atomically stores the `txid` in its `SparseChain<I>`
/// while also storing the transaction data in its `TxGraph`.
///
/// The `ChainGraph` does not guarantee any 1:1 mapping between transactions in the `chain` and
/// `graph` or vis versa. Both fields are public so they can mutated indepdendly. Even if you only
/// modify the `ChainGraph` through its atomic API, keep in mind that `TxGraph` does not allow
/// deletions while `SparseChain` does so deleting a transaction from the chain cannot delete it
/// from the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainGraph<I = TxHeight> {
    chain: SparseChain<I>,
    graph: TxGraph,
}

impl<I> Default for ChainGraph<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I> ChainGraph<I> {
    pub fn chain(&self) -> &sparse_chain::SparseChain<I> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }
}

impl<I: ChainIndex> ChainGraph<I> {
    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        index: Option<I>,
    ) -> Result<InsertOk, sparse_chain::InsertTxErr> {
        let chain_changed = match index {
            Some(index) => self.chain.insert_tx(tx.txid(), index)?,
            None => false,
        };
        let graph_changed = self.graph.insert_tx(tx);

        Ok(InsertOk {
            chain: chain_changed,
            graph: graph_changed,
        })
    }

    pub fn insert_output(&mut self, outpoint: OutPoint, txout: TxOut) -> bool {
        self.graph.insert_txout(outpoint, txout)
    }

    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<bool, sparse_chain::InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    /// Calculates the difference between self and `update` in the form of a [`ChangeSet`].
    pub fn determine_changeset(
        &self,
        update: &Self,
    ) -> Result<ChangeSet<I>, sparse_chain::UpdateFailure<I>> {
        let (mut chain_changeset, invalid_from) = self.chain.determine_changeset(&update.chain)?;
        let invalid_from: TxHeight = invalid_from.into();

        let conflicting_original_txids = update
            .chain
            .iter_txids()
            // skip txids that already exist in the original chain (for efficiency)
            .filter(|&(_, txid)| self.chain.tx_index(*txid).is_none())
            // skip txids that do not have full txs, as we can't check for conflicts for them
            .filter_map(|&(_, txid)| update.graph.tx(txid).or_else(|| self.graph.tx(txid)))
            // choose original txs that conflicts with the update
            .flat_map(|update_tx| {
                self.graph
                    .conflicting_txids(update_tx)
                    .filter_map(|(_, txid)| self.chain.tx_index(txid).map(|i| (txid, i)))
            });

        for (txid, original_index) in conflicting_original_txids {
            // if the evicted txid lies before "invalid_from", we screwed up
            if original_index.height() < invalid_from {
                return Err(sparse_chain::UpdateFailure::<I>::InconsistentTx {
                    inconsistent_txid: txid,
                    original_index: original_index.clone(),
                    update_index: None,
                });
            }

            chain_changeset.txids.insert(txid, None);
        }

        Ok(ChangeSet::<I> {
            chain: chain_changeset,
            graph: self.graph.determine_additions(&update.graph),
        })
    }

    /// Applies a [`ChangeSet`] to the chain graph
    pub fn apply_changeset(&mut self, changeset: ChangeSet<I>) {
        self.chain.apply_changeset(changeset.chain);
        self.graph.apply_additions(changeset.graph);
    }

    /// Applies the `update` chain graph. Note this is shorthand for calling [`determine_changeset`]
    /// and [`apply_changeset`] in sequence.
    ///
    /// [`apply_changeset`]: Self::apply_changeset
    /// [`determine_changeset`]: Self::determine_changeset
    pub fn apply_update(&mut self, update: Self) -> Result<(), sparse_chain::UpdateFailure<I>> {
        let changeset = self.determine_changeset(&update)?;
        self.apply_changeset(changeset);
        Ok(())
    }

    /// Get the full transaction output at an outpoint if it exists in the chain and the graph.
    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        self.chain.full_txout(&self.graph, outpoint)
    }

    /// Finds the transaction in the chain that spends `outpoint` given the input/output
    /// relationships in `graph`. Note that the transaction including `outpoint` does not need to be
    /// in the `graph` or the `chain` for this to return `Some(_)`.
    pub fn spent_by(&self, outpoint: OutPoint) -> Option<(&I, Txid)> {
        self.chain.spent_by(&self.graph, outpoint)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct ChangeSet<I> {
    pub chain: sparse_chain::ChangeSet<I>,
    pub graph: tx_graph::Additions,
}

impl<I> ChangeSet<I> {
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty() && self.graph.is_empty()
    }
}

impl<I> Default for ChangeSet<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I> AsRef<TxGraph> for ChainGraph<I> {
    fn as_ref(&self) -> &TxGraph {
        &self.graph
    }
}

impl<I> ForEachTxout for ChangeSet<I> {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.graph.for_each_txout(f)
    }
}

pub struct InsertOk<R = bool> {
    pub chain: R,
    pub graph: R,
}
