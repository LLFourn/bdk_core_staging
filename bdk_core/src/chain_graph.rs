use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    sparse_chain::{self, ChainIndex, SparseChain},
    tx_graph::TxGraph,
    BlockId, ConfirmationTime, TxHeight,
};

pub type TimestampedChainGraph = ChainGraph<ConfirmationTime>;

#[derive(Clone, Debug)]
pub struct ChainGraph<I: ChainIndex = TxHeight> {
    chain: SparseChain<I>,
    graph: TxGraph,
}

impl<I: ChainIndex> Default for ChainGraph<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I: sparse_chain::ChainIndex> ChainGraph<I> {
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

    pub fn chain(&self) -> &SparseChain<I> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn apply_update(
        &mut self,
        update: &Self,
    ) -> Result<sparse_chain::ChangeSet<I>, sparse_chain::UpdateFailure<I>> {
        let changeset = self.chain.determine_changeset(update.chain())?;
        changeset
            .tx_additions()
            .map(|new_txid| update.graph.tx(new_txid).expect("tx should exist"))
            .for_each(|tx| {
                self.graph.insert_tx(tx);
            });
        self.chain.apply_changeset(&changeset);
        Ok(changeset)
    }
}
