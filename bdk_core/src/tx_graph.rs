use bitcoin::{OutPoint, Transaction, Txid};

use crate::{alloc::vec::Vec, collections::*};

#[derive(Clone, Debug, Default)]
pub struct TxGraph {
    txs: HashMap<Txid, Transaction>, // Transaction enum? {Full|Partial}, or separate field?
    spends: BTreeMap<OutPoint, HashSet<Txid>>,
    // TODO: Aux outputs.
    // TODO: fn: Fee of tx
}

impl TxGraph {
    /// The outputs from the transaction with id `txid` that have been spent.
    pub fn outspend(&self, outpoint: &OutPoint) -> Option<&HashSet<Txid>> {
        self.spends.get(outpoint)
    }

    /// Each item contains the output index and the txid that spent that output.
    pub fn outspends(
        &self,
        txid: Txid,
    ) -> impl DoubleEndedIterator<Item = (u32, &HashSet<Txid>)> + '_ {
        let start = OutPoint { txid, vout: 0 };
        let end = OutPoint {
            txid,
            vout: u32::MAX,
        };
        self.spends
            .range(start..=end)
            .map(|(outpoint, spends)| (outpoint.vout, spends))
    }

    /// Get transaction by txid.
    pub fn tx(&self, txid: &Txid) -> Option<&Transaction> {
        self.txs.get(txid)
    }

    /// Add transaction, returns true when [`TxGraph`] is updated.
    pub fn insert_tx(&mut self, tx: &Transaction) -> bool {
        let txid = tx.txid();

        if self.txs.insert(txid, tx.clone()).is_some() {
            return false;
        }

        tx.input
            .iter()
            .map(|txin| txin.previous_output)
            .for_each(|outpoint| {
                self.spends
                    .entry(outpoint)
                    .or_insert_with(HashSet::new)
                    .insert(txid);
            });

        (0..tx.output.len() as u32)
            .map(|vout| OutPoint { txid, vout })
            .for_each(|outpoint| {
                self.spends.entry(outpoint).or_default();
            });

        return true;
    }

    pub fn is_unspent(&self, outpoint: &OutPoint) -> Option<bool> {
        self.spends.get(outpoint).map(|txids| txids.is_empty())
    }

    /// Return all txids of conflicting transactions to the given tx.
    pub fn conflicting_txids(&self, tx: &Transaction, include_self: bool) -> Vec<Txid> {
        let mut txids = tx
            .input
            .iter()
            .filter_map(|txin| self.spends.get(&txin.previous_output))
            .flat_map(|txids| txids.iter())
            .collect::<HashSet<_>>();
        if !include_self {
            txids.remove(&tx.txid());
        }
        txids.into_iter().cloned().collect()
    }
}
