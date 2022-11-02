use crate::{collections::*, Vec};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};

#[derive(Clone, Debug, Default)]
pub struct TxGraph {
    txs: HashMap<Txid, TxNode>,
    spends: BTreeMap<OutPoint, HashSet<Txid>>,
}

impl TxGraph {
    /// The outputs from the transaction with id `txid` that have been spent.
    pub fn outspend(&self, outpoint: OutPoint) -> Option<&HashSet<Txid>> {
        self.spends.get(&outpoint)
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

    /// Get a transaction by txid. This only returns `Some` for full transactions.
    pub fn tx(&self, txid: Txid) -> Option<&Transaction> {
        match self.txs.get(&txid)? {
            TxNode::Whole(tx) => Some(tx),
            TxNode::Partial(_) => None,
        }
    }

    /// Obtains a single tx output (if any) at specified outpoint.
    pub fn txout(&self, outpoint: OutPoint) -> Option<&TxOut> {
        match self.txs.get(&outpoint.txid)? {
            TxNode::Whole(tx) => tx.output.get(outpoint.vout as usize),
            TxNode::Partial(txouts) => txouts.get(&outpoint.vout),
        }
    }

    /// Returns a [`BTreeMap`] of outputs of a given txid.
    pub fn txouts(&self, txid: &Txid) -> Option<BTreeMap<u32, &TxOut>> {
        Some(match self.txs.get(txid)? {
            TxNode::Whole(tx) => tx
                .output
                .iter()
                .enumerate()
                .map(|(vout, txout)| (vout as u32, txout))
                .collect::<BTreeMap<_, _>>(),
            TxNode::Partial(txouts) => txouts
                .iter()
                .map(|(vout, txout)| (*vout, txout))
                .collect::<BTreeMap<_, _>>(),
        })
    }

    /// Add transaction, returns true when [`TxGraph`] is updated.
    pub fn insert_tx(&mut self, tx: &Transaction) -> bool {
        let txid = tx.txid();

        if let Some(TxNode::Whole(old_tx)) = self.txs.insert(txid, TxNode::Whole(tx.clone())) {
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
            .or_insert_with(TxNode::default);

        match tx_entry {
            TxNode::Whole(_) => false,
            TxNode::Partial(txouts) => txouts.insert(outpoint.vout as _, txout).is_some(),
        }
    }

    /// Calculates the fee of a given transaction (if we have all relevant data).
    pub fn calculate_fee(&self, tx: &Transaction) -> Option<u64> {
        let inputs_sum = tx
            .input
            .iter()
            .map(|txin| self.txout(txin.previous_output).map(|txout| txout.value))
            .sum::<Option<u64>>()?;

        let outputs_sum = tx.output.iter().map(|txout| txout.value).sum::<u64>();

        Some(
            inputs_sum
                .checked_sub(outputs_sum)
                .expect("tx graph has invalid data"),
        )
    }

    /// Iterate over all tx outputs known by [`TxGraph`].
    pub fn iter_all_txouts(&self) -> impl Iterator<Item = (OutPoint, &TxOut)> {
        self.txs.iter().flat_map(|(txid, tx)| match tx {
            TxNode::Whole(tx) => tx
                .output
                .iter()
                .enumerate()
                .map(|(vout, txout)| (OutPoint::new(*txid, vout as _), txout))
                .collect::<Vec<_>>(),
            TxNode::Partial(txouts) => txouts
                .iter()
                .map(|(vout, txout)| (OutPoint::new(*txid, *vout as _), txout))
                .collect::<Vec<_>>(),
        })
    }

    /// All the nodes in the graph
    pub fn nodes(&self) -> &HashMap<Txid, TxNode> {
        &self.txs
    }

    /// Return an iterator of conflicting txids, where the first field of the tuple is the vin of
    /// the original tx in which the txid conflicts.
    pub fn conflicting_txids<'g>(
        &'g self,
        tx: &'g Transaction,
    ) -> impl Iterator<Item = (usize, Txid)> + '_ {
        tx.input
            .iter()
            .enumerate()
            .flat_map(|(vin, txin)| {
                self.spends
                    .get(&txin.previous_output)
                    .into_iter()
                    .flat_map(|spend_set| spend_set.iter())
                    .map(move |&spend_txid| (vin, spend_txid))
            })
            .filter(move |(_, spend_txid)| spend_txid != &tx.txid())
    }

    /// Extends this graph with another so that `self` becomes the union of the two sets of
    /// transactions.
    pub fn extend(&mut self, other: TxGraph) {
        for (txid, tx) in other.txs {
            match tx {
                TxNode::Whole(tx) => {
                    self.insert_tx(&tx);
                }
                TxNode::Partial(partial) => {
                    for (vout, txout) in partial {
                        self.insert_txout(OutPoint { txid, vout }, txout);
                    }
                }
            }
        }
    }
}

/// Node of a [`TxGraph`]
#[derive(Clone, Debug)]
pub enum TxNode {
    Whole(Transaction),
    Partial(BTreeMap<u32, TxOut>),
}

impl Default for TxNode {
    fn default() -> Self {
        Self::Partial(BTreeMap::new())
    }
}
