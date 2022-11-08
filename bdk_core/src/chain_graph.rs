use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    BlockId, ChangeSet, InsertCheckpointErr, InsertTxErr, SparseChain, TxData, TxGraph, TxHeight,
    UpdateFailure,
};

#[derive(Clone, Debug, Default)]
pub struct ChainGraph<D = ()> {
    chain: SparseChain<D>,
    graph: TxGraph,
}

impl<D: Clone + Debug + Default + Ord> ChainGraph<D> {
    pub fn insert_tx(&mut self, tx: Transaction, height: TxHeight) -> Result<bool, InsertTxErr> {
        self.insert_tx_with_additional_data(tx, height.into())
    }

    pub fn insert_tx_with_additional_data(
        &mut self,
        tx: Transaction,
        additional_data: TxData<D>,
    ) -> Result<bool, InsertTxErr> {
        let changed = self
            .chain
            .insert_tx_with_additional_data(tx.txid(), additional_data)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        height: TxHeight,
    ) -> Result<bool, InsertTxErr> {
        self.insert_output_with_additional_data(outpoint, txout, height.into())
    }

    pub fn insert_output_with_additional_data(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        additional_data: TxData<D>,
    ) -> Result<bool, InsertTxErr> {
        let changed = self
            .chain
            .insert_tx_with_additional_data(outpoint.txid, additional_data)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(&mut self, txid: Txid, height: TxHeight) -> Result<bool, InsertTxErr> {
        self.insert_txid_with_additional_data(txid, height.into())
    }

    pub fn insert_txid_with_additional_data(
        &mut self,
        txid: Txid,
        additional_data: TxData<D>,
    ) -> Result<bool, InsertTxErr> {
        self.chain
            .insert_tx_with_additional_data(txid, additional_data)
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
