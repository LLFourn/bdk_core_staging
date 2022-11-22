use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    sparse_chain::{self, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ChainIndex, ConfirmationTime, ForEachTxout, FullTxOut, TxHeight,
};

pub type TimestampedChainGraph = ChainGraph<ConfirmationTime>;

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
    pub fn chain(&self) -> &SparseChain<I> {
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
        index: I,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), index)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(&mut self, outpoint: OutPoint, txout: TxOut) -> bool {
        self.graph.insert_txout(outpoint, txout)
    }

    pub fn insert_txid(&mut self, txid: Txid, index: I) -> Result<bool, sparse_chain::InsertTxErr> {
        self.chain.insert_tx(txid, index)
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
        Ok(ChangeSet::<I> {
            chain: self.chain.determine_changeset(&update.chain)?,
            graph: self.graph.determine_additions(&update.graph),
        })
    }

    /// Applies a [`ChangeSet`] to the chain graph
    pub fn apply_changeset(&mut self, changeset: &ChangeSet<I>) {
        self.chain.apply_changeset(&changeset.chain);
        self.graph.apply_additions(&changeset.graph);
    }

    /// Applies the `update` chain graph. Note this is shorthand for calling [`determine_changeset`]
    /// and [`apply_changeset`] in sequence.
    pub fn apply_update(
        &mut self,
        update: &Self,
    ) -> Result<ChangeSet<I>, sparse_chain::UpdateFailure<I>> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok(changeset)
    }

    /// Get the full transaction output at an outpoint if it exists in the chain and the graph.
    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        self.chain().full_txout(self.graph(), outpoint)
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
        self.graph()
    }
}

impl<I> ForEachTxout for ChangeSet<I> {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.graph.for_each_txout(f)
    }
}
