use bitcoin::{OutPoint, Transaction, TxOut, Txid};

use crate::{
    BlockId, ChainIndex, ChangeSet, InsertCheckpointErr, InsertTxErr, SparseChain, TxGraph,
    UpdateFailure,
};

#[derive(Clone, Debug, Default)]
pub struct ChainGraph<I> {
    chain: SparseChain<I>,
    graph: TxGraph,
}

impl<I> ChainGraph<I>
where
    I: ChainIndex,
{
    pub fn insert_tx(&mut self, tx: Transaction, chain_index: I) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), chain_index)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        chain_index: I,
    ) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, chain_index)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(&mut self, txid: Txid, chain_index: I) -> Result<bool, InsertTxErr> {
        self.chain.insert_tx(txid, chain_index)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &SparseChain<I> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let changeset = self.chain.determine_changeset(update.chain())?;
        changeset
            .tx_additions()
            .map(|new_txid| update.graph.tx(new_txid).expect("tx should exist"))
            .for_each(|tx| {
                self.graph.insert_tx(tx);
            });
        self.chain.apply_changeset(changeset.clone());
        Ok(changeset)
    }
}
