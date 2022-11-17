use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::{Debug, Pointer};

use crate::{
    collections::HashSet,
    sparse_chain::{self, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ChainIndex, ConfirmationTime, TxHeight, Vec,
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

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: &TxOut,
        index: I,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, index)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
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

    pub fn determine_changeset(&self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let chain_changeset = self.chain.determine_changeset(&update.chain)?;
        let graph_additions = self.graph.determine_additions(&update.graph);

        let added_txids = graph_additions.txids().collect::<HashSet<_>>();

        // ensure changeset adds exist in either graph_additions or self.graph
        let missing = chain_changeset
            .tx_additions()
            .filter(|txid| added_txids.contains(txid) || self.graph.contains_txid(*txid))
            .collect::<HashSet<_>>();
        if !missing.is_empty() {
            return Err(UpdateFailure::Missing(missing));
        }

        Ok(ChangeSet::<I> {
            chain: chain_changeset,
            graph: graph_additions,
        })
    }

    pub fn apply_changeset(&mut self, changeset: &ChangeSet<I>) {
        self.chain.apply_changeset(&changeset.chain);
        self.graph.apply_additions(&changeset.graph);
    }

    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let changeset = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok(changeset)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct ChangeSet<I> {
    chain: sparse_chain::ChangeSet<I>,
    graph: tx_graph::Additions,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure<I> {
    Chain(sparse_chain::UpdateFailure<I>),
    Missing(HashSet<Txid>),
}

impl<I> core::fmt::Display for UpdateFailure<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UpdateFailure::Chain(err) => err.fmt(f),
            UpdateFailure::Missing(txid) => write!(f, "missing txs: {:?}", txid),
        }
    }
}

#[cfg(feature = "std")]
impl<I: core::fmt::Debug> std::error::Error for UpdateFailure<I> {}

impl<I> From<sparse_chain::UpdateFailure<I>> for UpdateFailure<I> {
    fn from(err: sparse_chain::UpdateFailure<I>) -> Self {
        Self::Chain(err)
    }
}
