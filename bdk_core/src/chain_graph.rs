use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    BlockId, ChainIndex, ChainIndexExtension, ChangeSet, InsertCheckpointErr, InsertTxErr,
    SparseChain, TxGraph, UpdateFailure,
};

pub type TimestampedChainGraph = ChainGraph<Option<u64>>;

#[derive(Clone, Debug, Default)]
pub struct ChainGraph<E = ()> {
    chain: SparseChain<E>,
    graph: TxGraph,
}

impl<E: ChainIndexExtension> ChainGraph<E> {
    pub fn insert_tx<I>(&mut self, tx: Transaction, index: I) -> Result<bool, InsertTxErr>
    where
        I: Into<ChainIndex<E>>,
    {
        let changed = self.chain.insert_tx(tx.txid(), index)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output<I>(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        index: I,
    ) -> Result<bool, InsertTxErr>
    where
        I: Into<ChainIndex<E>>,
    {
        let changed = self.chain.insert_tx(outpoint.txid, index)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid<I>(&mut self, txid: Txid, index: I) -> Result<bool, InsertTxErr>
    where
        I: Into<ChainIndex<E>>,
    {
        self.chain.insert_tx(txid, index)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &SparseChain<E> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<E>, UpdateFailure<E>> {
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
