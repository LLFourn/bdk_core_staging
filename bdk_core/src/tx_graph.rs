use bitcoin::{OutPoint, Transaction, TxIn, Txid};

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
                self.spends.entry(outpoint).or_default().insert(txid);
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

    /// Returns all txids of conflicting spends.
    pub fn conflicting_spends<'g, 't>(
        &'g self,
        txid: &'t Txid,
        txin: &TxIn,
    ) -> impl Iterator<Item = &'g Txid> + 't
    where
        'g: 't,
    {
        self.spends
            .get(&txin.previous_output)
            .into_iter()
            .flat_map(|spend_set| spend_set.iter())
            .filter(move |&spend_txid| spend_txid != txid)
    }

    /// Return an iterator of conflicting txids, where the first field of the tuple is the vin of
    /// the original tx in which the txid conflicts.
    pub fn conflicting_txids<'g, 't>(
        &'g self,
        tx: &'t Transaction,
    ) -> impl Iterator<Item = (usize, &'g Txid)> + 't
    where
        'g: 't,
    {
        tx.input.iter().enumerate().flat_map(|(vin, txin)| {
            self.conflicting_spends(&tx.txid(), txin)
                .map(move |spend_txid| (vin, spend_txid))
                .collect::<Vec<_>>()
        })
    }
}
