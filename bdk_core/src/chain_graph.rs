use bitcoin::{OutPoint, Transaction, TxOut, Txid};

use crate::{BlockId, InsertCheckpointErr, InsertTxErr, SparseChain, TxGraph, TxHeight};

#[derive(Clone, Debug, Default)]
pub struct ChainGraph {
    chain: SparseChain,
    graph: TxGraph,
}

impl ChainGraph {
    pub fn insert_tx(&mut self, tx: Transaction, height: TxHeight) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), height)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        height: TxHeight,
    ) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, height)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(&mut self, txid: Txid, height: TxHeight) -> Result<bool, InsertTxErr> {
        self.chain.insert_tx(txid, height)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &SparseChain {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }
}
