use bitcoin::{OutPoint, Transaction, TxOut, Txid};

use crate::{
    BlockId, ChangeSet, InsertCheckpointErr, InsertTxErr, SparseChain, TxData, TxGraph,
    UpdateFailure,
};

#[derive(Clone, Debug, Default)]
pub struct ChainGraph<D> {
    chain: SparseChain<D>,
    graph: TxGraph,
}

impl<D: Clone + core::fmt::Debug + Default + Ord> ChainGraph<D> {
    pub fn insert_tx(&mut self, tx: Transaction, data: TxData<D>) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), data)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        data: TxData<D>,
    ) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, data)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(&mut self, txid: Txid, data: TxData<D>) -> Result<bool, InsertTxErr> {
        self.chain.insert_tx(txid, data)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &SparseChain<D> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<D>, UpdateFailure> {
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
