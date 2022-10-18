use bitcoin::{OutPoint, Transaction, TxIn, TxOut, Txid};

use crate::{alloc::vec::Vec, collections::*, Box};

#[derive(Clone, Debug, Default)]
pub struct TxGraph {
    txs: HashMap<Txid, TxInGraph>,
    spends: BTreeMap<OutPoint, HashSet<Txid>>,
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
    pub fn tx(&self, txid: &Txid) -> Option<&TxInGraph> {
        self.txs.get(txid)
    }

    /// Add transaction, returns true when [`TxGraph`] is updated.
    pub fn insert_tx(&mut self, tx: &Transaction) -> bool {
        let txid = tx.txid();

        if let Some(TxInGraph::Whole(old_tx)) = self.txs.insert(txid, TxInGraph::Whole(tx.clone()))
        {
            debug_assert_eq!(&old_tx, tx);
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

    /// Inserts an auxiliary txout. Returns false if txout already exists.
    pub fn insert_txout(&mut self, outpoint: OutPoint, txout: TxOut) -> bool {
        let tx_entry = self
            .txs
            .entry(outpoint.txid)
            .or_insert_with(TxInGraph::default);

        match tx_entry {
            TxInGraph::Whole(_) => false,
            TxInGraph::Partial(txouts) => txouts.insert(outpoint.vout as _, txout).is_some(),
        }
    }

    /// Determines whether outpoint is spent or not. Returns `None` when outpoint does not exist in
    /// graph.
    pub fn is_unspent(&self, outpoint: &OutPoint) -> Option<bool> {
        self.spends.get(outpoint).map(|txids| txids.is_empty())
    }

    /// Iterate over all txouts known by [`TxGraph`].
    pub fn iter_txout<'g>(&'g self) -> impl Iterator<Item = (OutPoint, &'g TxOut)> {
        self.txs.iter().flat_map(|(txid, tx)| {
            tx.iter_outputs().map(|(vout, txout)| {
                let op = OutPoint {
                    txid: *txid,
                    vout: vout as u32,
                };
                (op, txout)
            })
        })
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

/// Transaction can either be whole, or we may only have auxiliary txouts avaliable.
#[derive(Clone, Debug)]
pub enum TxInGraph {
    Whole(Transaction),
    Partial(BTreeMap<usize, TxOut>),
}

impl Default for TxInGraph {
    fn default() -> Self {
        Self::Partial(BTreeMap::new())
    }
}

impl TxInGraph {
    pub fn output<'t>(&'t self, vout: usize) -> Option<&'t TxOut> {
        match self {
            TxInGraph::Whole(tx) => tx.output.get(vout),
            TxInGraph::Partial(txouts) => txouts.get(&vout),
        }
    }

    pub fn iter_outputs<'t>(
        &'t self,
    ) -> Box<dyn DoubleEndedIterator<Item = (usize, &'t TxOut)> + 't> {
        match self {
            TxInGraph::Whole(tx) => Box::new(tx.output.iter().enumerate()),
            TxInGraph::Partial(txouts) => {
                Box::new(txouts.iter().map(|(vout, txout)| (*vout, txout)))
            }
        }
    }
}
