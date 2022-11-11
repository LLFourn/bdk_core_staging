use crate::{sparse_chain, tx_graph, BlockId, TxHeight};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChainGraph {
    chain: sparse_chain::SparseChain,
    graph: tx_graph::TxGraph,
}

impl ChainGraph {
    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        height: TxHeight,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), height)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        height: TxHeight,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, height)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(
        &mut self,
        txid: Txid,
        height: TxHeight,
    ) -> Result<bool, sparse_chain::InsertTxErr> {
        self.chain.insert_tx(txid, height)
    }

    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<bool, sparse_chain::InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &sparse_chain::SparseChain {
        &self.chain
    }

    pub fn graph(&self) -> &tx_graph::TxGraph {
        &self.graph
    }

    pub fn determine_changeset(
        &self,
        update: &Self,
    ) -> Result<ChangeSet, sparse_chain::UpdateFailure> {
        let chain_changeset = self.chain.determine_changeset(update.chain())?;
        let graph_additions = self.graph.determine_additions(update.graph());
        Ok(ChangeSet {
            chain: chain_changeset.into_new_change_set(),
            graph: graph_additions,
        })
    }

    pub fn apply_update(&mut self, update: Self) -> Result<(), sparse_chain::UpdateFailure> {
        let changeset = self.determine_changeset(&update)?;
        self.apply_changeset(changeset);
        Ok(())
    }

    pub fn apply_changeset(&mut self, changeset: ChangeSet) {
        self.chain.apply_changeset(changeset.chain);
        self.graph.apply_additions(changeset.graph);
    }
}

#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeSet {
    pub chain: sparse_chain::NewChangeSet,
    pub graph: tx_graph::Additions,
}

impl ChangeSet {
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty() && self.graph.is_empty()
    }
}
