use crate::{collections::*, ForEachTxout};
use alloc::{borrow::Cow, vec::Vec};
use bitcoin::{OutPoint, Transaction, TxIn, TxOut, Txid};

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

/// Do not filter any descendants.
///
/// This is a convenience method intended for use with [`TxGraph::iter_tx_descendants()`] and
/// related methods.
pub fn no_filter(_: usize, _: Txid) -> bool {
    true
}

impl TxGraph {
    /// The transactions spending from this output.
    ///
    /// `TxGraph` allows conflicting transactions within the graph. Obviously the transactions in
    /// the returned will never be in the same blockchain.
    ///
    /// Note this returns a [`Cow`] because of an implementation detail.
    ///
    /// [`Cow`]: std::borrow::Cow
    // FIXME: this Cow could be gotten rid of if we could do HashSet::new in a const fn
    pub fn outspends(&self, outpoint: OutPoint) -> Cow<HashSet<Txid>> {
        self.spends
            .get(&outpoint)
            .map(|outspends| Cow::Borrowed(outspends))
            .unwrap_or(Cow::Owned(HashSet::default()))
    }

    /// The transactions spending from `txid`.
    pub fn tx_outspends(
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
    pub fn get_tx(&self, txid: Txid) -> Option<&Transaction> {
        match self.txs.get(&txid)? {
            TxNode::Whole(tx) => Some(tx),
            TxNode::Partial(_) => None,
        }
    }

    /// Obtains a single tx output (if any) at specified outpoint.
    pub fn get_txout(&self, outpoint: OutPoint) -> Option<&TxOut> {
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

    /// Returns the resultant [`Additions`] if the given transaction is inserted. Does not actually
    /// mutate [`Self`].
    pub fn insert_tx_preview(&self, tx: Transaction) -> Additions {
        let mut update = Self::default();
        update.txs.insert(tx.txid(), TxNode::Whole(tx));
        self.determine_additions(&update)
    }

    /// Inserts the given transaction into [`Self`].
    pub fn insert_tx(&mut self, tx: Transaction) -> Additions {
        let additions = self.insert_tx_preview(tx);
        self.apply_additions(additions.clone());
        additions
    }

    /// Returns the resultant [`Additions`] if the given `txout` is inserted at `outpoint`. Does not
    /// mutate `self`.
    pub fn insert_txout_preview(&self, outpoint: OutPoint, txout: TxOut) -> Additions {
        let mut update = Self::default();
        update.txs.insert(
            outpoint.txid,
            TxNode::Partial([(outpoint.vout, txout)].into()),
        );
        self.determine_additions(&update)
    }

    /// Inserts the given [`TxOut`] at [`OutPoint`].
    ///
    /// Note this will ignore the action if we already have the full transaction that the txout is
    /// alledged to be on (even if it doesn't match it!).
    pub fn insert_txout(&mut self, outpoint: OutPoint, txout: TxOut) -> Additions {
        let additions = self.insert_txout_preview(outpoint, txout);
        self.apply_additions(additions.clone());
        additions
    }

    /// Calculates the fee of a given transaction. Returns 0 if `tx` is a coinbase transaction.
    /// Returns `Some(_)` if we have all the `TxOut`s being spent by `tx` in the graph (either as
    /// the full transactions or individual txouts). If the returned value is negative then the
    /// transaction is invalid according to the graph.
    ///
    /// Returns `None` if we're missing an input for the tx in the graph.
    ///
    /// Note `tx` does not have to be in the graph for this to work.
    pub fn calculate_fee(&self, tx: &Transaction) -> Option<i64> {
        if tx.is_coin_base() {
            return Some(0);
        }
        let inputs_sum = tx
            .input
            .iter()
            .map(|txin| {
                self.get_txout(txin.previous_output)
                    .map(|txout| txout.value as i64)
            })
            .sum::<Option<i64>>()?;

        let outputs_sum = tx
            .output
            .iter()
            .map(|txout| txout.value as i64)
            .sum::<i64>();

        Some(inputs_sum - outputs_sum)
    }

    /// Iterate over all tx outputs known by [`TxGraph`].
    pub fn all_txouts(&self) -> impl Iterator<Item = (OutPoint, &TxOut)> {
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

    /// Iterate over all full transactions in the graph.
    pub fn full_transactions(&self) -> impl Iterator<Item = &Transaction> {
        self.txs.iter().filter_map(|(_, tx)| match tx {
            TxNode::Whole(tx) => Some(tx),
            TxNode::Partial(_) => None,
        })
    }

    /// Iterate over all partial transactions (outputs only) in the graph.
    pub fn partial_transactions(&self) -> impl Iterator<Item = (Txid, &BTreeMap<u32, TxOut>)> {
        self.txs.iter().filter_map(|(txid, tx)| match tx {
            TxNode::Whole(_) => None,
            TxNode::Partial(partial) => Some((*txid, partial)),
        })
    }

    /// Iterates descendant transactions of given `txid` (including the given `txid`).
    ///
    /// if `path_filter` returns `true`, we will continue exploring the path from the `txid` of
    /// `depth`, where `depth: usize` is the distance from the initial `txid` to the descendant.
    pub fn iter_tx_descendants<'g>(
        &'g self,
        txid: Txid,
        path_filter: impl FnMut(usize, Txid) -> bool + 'g,
    ) -> impl Iterator<Item = (usize, Txid)> + '_ {
        TxDescendants::new(self, txid, path_filter)
    }

    /// Iterates over all conflicting txids (including descendants) of the given transaction.
    ///
    /// The `path_filter` filters descendant paths to check. Refer to
    /// [`Self::iter_tx_descendants()`] for more details.
    pub fn tx_conflicts<'g>(
        &'g self,
        tx: &'g Transaction,
        path_filter: impl Fn(usize, Txid) -> bool + 'g,
    ) -> impl Iterator<Item = (usize, Txid)> + '_ {
        self.adjacent_conflicts_of_tx(tx)
            .flat_map(move |(_, txid)| {
                self.iter_tx_descendants(txid, &path_filter)
                    .collect::<Vec<_>>()
            })
    }

    /// Given a `txin` and the `txid` of the transaction in which it resides, return an iterator of
    /// txids which directly conflict with the given `txin`.
    ///
    /// **WARNING:** It is up to the caller to ensure the given `txin` actually resides in `txid`.
    /// **WARNING:** This only returns directly conflicting txids and does not include descendants
    /// of those txids.
    pub fn adjacent_conflicts_of_txin<'g>(
        &'g self,
        txid: Txid,
        txin: &'g TxIn,
    ) -> impl Iterator<Item = Txid> + '_ {
        self.spends
            .get(&txin.previous_output)
            .into_iter()
            .flatten()
            .filter(move |&&conflicting_txid| conflicting_txid != txid)
            .cloned()
    }

    /// Given a transaction, return an iterator of txids which directly conflict with the given
    /// transaction's inputs (spends). The conflicting txids are returned with the given
    /// transaction's vin (in which it conflicts).
    ///
    /// **WARNING:** This only returns directly conflicting txids and does not include descendants
    /// of those txids.
    pub fn adjacent_conflicts_of_tx<'g>(
        &'g self,
        tx: &'g Transaction,
    ) -> impl Iterator<Item = (usize, Txid)> + '_ {
        let txid = tx.txid();
        tx.input.iter().enumerate().flat_map(move |(vin, txin)| {
            self.adjacent_conflicts_of_txin(txid, txin)
                .map(move |txid| (vin, txid))
        })
    }

    /// Previews the resultant [`Additions`] when [`Self`] is updated against the `update` graph.
    pub fn determine_additions(&self, update: &Self) -> Additions {
        let mut additions = Additions::default();

        for (&txid, update_tx) in &update.txs {
            if self.get_tx(txid).is_some() {
                continue;
            }

            match update_tx {
                TxNode::Whole(tx) => {
                    if matches!(self.txs.get(&txid), None | Some(TxNode::Partial(_))) {
                        additions.tx.insert(tx.clone());
                    }
                }
                TxNode::Partial(partial) => {
                    for (&vout, update_txout) in partial {
                        let outpoint = OutPoint::new(txid, vout);

                        if self.get_txout(outpoint) != Some(&update_txout) {
                            additions.txout.insert(outpoint, update_txout.clone());
                        }
                    }
                }
            }
        }

        additions
    }

    /// Extends this graph with another so that `self` becomes the union of the two sets of
    /// transactions.
    pub fn apply_update(&mut self, update: TxGraph) -> Additions {
        let additions = self.determine_additions(&update);
        self.apply_additions(additions.clone());
        additions
    }

    pub fn apply_additions(&mut self, additions: Additions) {
        for tx in additions.tx {
            let txid = tx.txid();

            tx.input
                .iter()
                .map(|txin| txin.previous_output)
                // coinbase spends are not to be counted
                .filter(|outpoint| !outpoint.is_null())
                // record spend as this tx has spent this outpoint
                .for_each(|outpoint| {
                    self.spends.entry(outpoint).or_default().insert(txid);
                });

            if let Some(TxNode::Whole(old_tx)) = self.txs.insert(txid, TxNode::Whole(tx)) {
                debug_assert_eq!(
                    old_tx.txid(),
                    txid,
                    "old tx of same txid should not be different"
                );
            }
        }

        for (outpoint, txout) in additions.txout {
            let tx_entry = self
                .txs
                .entry(outpoint.txid)
                .or_insert_with(TxNode::default);

            match tx_entry {
                TxNode::Whole(_) => { /* do nothing since we already have full tx */ }
                TxNode::Partial(txouts) => {
                    txouts.insert(outpoint.vout, txout);
                }
            }
        }
    }

    /// Whether the graph has any transactions or outputs in it.
    pub fn is_empty(&self) -> bool {
        self.txs.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
#[must_use]
pub struct Additions {
    pub tx: BTreeSet<Transaction>,
    pub txout: BTreeMap<OutPoint, TxOut>,
}

impl Additions {
    pub fn is_empty(&self) -> bool {
        self.tx.is_empty() && self.txout.is_empty()
    }

    /// Iterates over [`Txid`]s mentioned in [`Additions`], whether they be full txs (`true`) or
    /// individual outputs (`false`).
    ///
    /// This does not guarantee that there will not be duplicate txids.
    pub fn txids(&self) -> impl Iterator<Item = (Txid, bool)> + '_ {
        let partials = self.txout.keys().map(|op| (op.txid, false));
        let fulls = self.tx.iter().map(|tx| (tx.txid(), true));

        partials.chain(fulls)
    }

    pub fn txouts(&self) -> impl Iterator<Item = (OutPoint, &TxOut)> {
        self.tx
            .iter()
            .flat_map(|tx| {
                tx.output
                    .iter()
                    .enumerate()
                    .map(|(vout, txout)| (OutPoint::new(tx.txid(), vout as _), txout))
            })
            .chain(self.txout.iter().map(|(op, txout)| (*op, txout)))
    }

    /// Appends the changes in `other` into self such that applying `self` afterwards has the same
    /// effect as sequentially applying the original `self` and `other`.
    pub fn append(&mut self, mut other: Additions) {
        self.tx.append(&mut other.tx);
        self.txout.append(&mut other.txout);
    }
}

impl<T: AsRef<TxGraph>> ForEachTxout for T {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.as_ref().all_txouts().for_each(f)
    }
}

impl AsRef<TxGraph> for TxGraph {
    fn as_ref(&self) -> &TxGraph {
        self
    }
}

impl ForEachTxout for Additions {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.txouts().for_each(f)
    }
}

pub struct TxDescendants<'a, F> {
    graph: &'a TxGraph,
    path_filter: F,
    visited: HashSet<Txid>,
    stack: Vec<(usize, Txid)>,
}

impl<'a, F> TxDescendants<'a, F> {
    fn new(graph: &'a TxGraph, txid: Txid, path_filter: F) -> Self {
        Self {
            graph,
            path_filter,
            visited: [].into(),
            stack: [(0, txid)].into(),
        }
    }
}

impl<'a, F> Iterator for TxDescendants<'a, F>
where
    F: FnMut(usize, Txid) -> bool,
{
    type Item = (usize, Txid);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (depth, txid) = self.stack.pop()?;
            if !self.visited.insert(txid) {
                continue;
            }
            if (self.path_filter)(depth, txid) {
                let mut children = self
                    .graph
                    .spends
                    .range(OutPoint::new(txid, u32::MIN)..=OutPoint::new(txid, u32::MAX))
                    .flat_map(|(_, txids)| txids)
                    .map(|&txid| (depth + 1, txid))
                    .collect::<Vec<_>>();
                self.stack.append(&mut children);
                return Some((depth, txid));
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.stack.len(), None)
    }
}
