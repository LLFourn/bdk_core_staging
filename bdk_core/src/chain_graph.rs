use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    collections::HashMap,
    sparse_chain::{self, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ChainIndex, ConfirmationTime, TxHeight,
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
        txout: TxOut,
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

    /// Calculates the difference between self and `update` in the form of a [`ChangeSet`].
    ///
    /// Additionally, missing transactions are returned in the form of [`HashMap<Txid, bool>`],
    /// where the value is `true` for partially missing, and `false` for fully missing.
    pub fn determine_changeset(
        &self,
        update: &Self,
    ) -> Result<(ChangeSet<I>, HashMap<Txid, bool>), sparse_chain::UpdateFailure<I>> {
        let chain_changeset = self.chain.determine_changeset(&update.chain)?;
        let graph_additions = self.graph.determine_additions(&update.graph);

        let added_txids = graph_additions.txids().collect::<HashMap<_, _>>();

        let missing = chain_changeset
            .tx_additions()
            .filter_map(|txid| match added_txids.get(&txid).cloned() {
                Some(true) => None,                // not missing
                Some(false) => Some((txid, true)), // partially missing
                None => Some((txid, false)),       // completely missing
            })
            .collect::<HashMap<_, _>>();

        Ok((
            ChangeSet::<I> {
                chain: chain_changeset,
                graph: graph_additions,
            },
            missing,
        ))
    }

    pub fn apply_changeset(&mut self, changeset: &ChangeSet<I>) {
        self.chain.apply_changeset(&changeset.chain);
        self.graph.apply_additions(&changeset.graph);
    }

    pub fn apply_update(
        &mut self,
        update: &Self,
    ) -> Result<(ChangeSet<I>, HashMap<Txid, bool>), sparse_chain::UpdateFailure<I>> {
        let (changeset, missing) = self.determine_changeset(update)?;
        self.apply_changeset(&changeset);
        Ok((changeset, missing))
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
