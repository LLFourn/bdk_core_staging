use bitcoin::{OutPoint, Transaction, TxOut, Txid};

use crate::{collections::*, Vec};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TxGraph {
    txs: HashMap<Txid, TxNode>,
    spends: BTreeMap<OutPoint, HashSet<Txid>>,
}

/// Node of a [`TxGraph`]
#[derive(Clone, Debug, PartialEq)]
enum TxNode {
    Whole(Transaction),
    Partial(BTreeMap<u32, TxOut>),
}

impl Default for TxNode {
    fn default() -> Self {
        Self::Partial(BTreeMap::new())
    }
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

    /// Returns true when graph contains given tx of txid (whether it be partial or full).
    pub fn contains_txid(&self, txid: Txid) -> bool {
        self.txs.contains_key(&txid)
    }

    /// Obtains a single tx output (if any) at specified outpoint.
    pub fn txout(&self, outpoint: OutPoint) -> Option<&TxOut> {
        match self.txs.get(&outpoint.txid)? {
            TxNode::Whole(tx) => tx.output.get(outpoint.vout as usize),
            TxNode::Partial(txouts) => txouts.get(&outpoint.vout),
        }
    }

    /// Returns a [`BTreeMap`] of outputs of a given txid.
    pub fn txouts(&self, txid: Txid) -> Option<BTreeMap<u32, &TxOut>> {
        Some(match self.txs.get(&txid)? {
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

    /// Inserts an auxiliary txout. Returns true if txout is newly added.
    pub fn insert_txout(&mut self, outpoint: OutPoint, txout: &TxOut) -> bool {
        let tx_entry = self
            .txs
            .entry(outpoint.txid)
            .or_insert_with(TxNode::default);

        match tx_entry {
            TxNode::Whole(_) => false,
            TxNode::Partial(txouts) => match txouts.insert(outpoint.vout as _, txout.clone()) {
                Some(old_txout) => {
                    debug_assert_eq!(txout, &old_txout);
                    false
                }
                None => true,
            },
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

    /// Iterate over all full transactions in the graph
    pub fn iter_full_transactions(&self) -> impl Iterator<Item = &Transaction> {
        self.txs.iter().filter_map(|(_, tx)| match tx {
            TxNode::Whole(tx) => Some(tx),
            TxNode::Partial(_) => None,
        })
    }

    pub fn iter_partial_transactions(&self) -> impl Iterator<Item = (Txid, &BTreeMap<u32, TxOut>)> {
        self.txs.iter().filter_map(|(txid, tx)| match tx {
            TxNode::Whole(_) => None,
            TxNode::Partial(partial) => Some((*txid, partial)),
        })
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
    pub fn apply_update(&mut self, update: &TxGraph) -> Additions {
        let additions = self.determine_additions(update);
        self.apply_additions(&additions);
        additions
    }

    pub fn determine_additions(&self, update: &TxGraph) -> Additions {
        let mut additions = Additions::default();

        for (&txid, tx) in &update.txs {
            match tx {
                TxNode::Whole(tx) => {
                    if self.tx(txid).is_none() {
                        additions.tx.insert(tx.clone());
                    }
                }
                TxNode::Partial(partial) => {
                    for (&vout, txout) in partial {
                        let op = OutPoint { txid, vout };
                        let insert = match self.txouts(txid) {
                            Some(txouts) => match txouts.get(&vout) {
                                Some(existing_txout) => *existing_txout != txout,
                                None => true,
                            },
                            None => true,
                        };

                        if insert {
                            additions.txout.insert(op, txout.clone());
                        }
                    }
                }
            }
        }

        additions
    }

    pub fn apply_additions(&mut self, additions: &Additions) {
        for tx in &additions.tx {
            self.insert_tx(tx);
        }

        for (outpoint, txout) in &additions.txout {
            self.insert_txout(*outpoint, txout);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct Additions {
    pub tx: BTreeSet<Transaction>,
    pub txout: BTreeMap<OutPoint, TxOut>,
}

impl Additions {
    pub fn is_empty(&self) -> bool {
        self.tx.is_empty() && self.txout.is_empty()
    }

    pub fn txids<B: core::iter::FromIterator<Txid>>(&self) -> B {
        self.tx
            .iter()
            .map(Transaction::txid)
            .chain(self.txout.keys().map(|op| op.txid))
            .collect()
    }
}
