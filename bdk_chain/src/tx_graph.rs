use crate::{collections::*, ForEachTxout};
use alloc::{borrow::Cow, vec::Vec};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};

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

    /// Returns the resultant [`Additions`] if the given transaction is inserted. Does not actually
    /// mutate [`Self`].
    pub fn insert_tx_preview(&mut self, tx: Transaction) -> Additions {
        let mut update = Self::default();
        update.txs.insert(tx.txid(), TxNode::Whole(tx));

        self.apply_update(update)
            .expect("txout should be as expected")
    }

    /// Inserts the given transaction into [`Self`].
    pub fn insert_tx(&mut self, tx: Transaction) -> Additions {
        let additions = self.insert_tx_preview(tx);
        self.apply_additions(additions.clone());
        additions
    }

    /// Returns the resultant [`Additions`] if the given [`TxOut`] is inserted at [`OutPoint`]. Does
    /// not actually mutate [`Self`].
    pub fn insert_txout_preview(
        &self,
        outpoint: OutPoint,
        txout: TxOut,
    ) -> Result<Additions, TxOutConflict> {
        let mut update = Self::default();
        update.txs.insert(
            outpoint.txid,
            TxNode::Partial([(outpoint.vout, txout)].into()),
        );
        self.determine_additions(&update)
    }

    /// Inserts the given [`TxOut`] at [`OutPoint`].
    pub fn insert_txout(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
    ) -> Result<Additions, TxOutConflict> {
        let additions = self.insert_txout_preview(outpoint, txout)?;
        self.apply_additions(additions.clone());
        Ok(additions)
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

    /// Iterate over all full transactions in the graph
    pub fn full_transactions(&self) -> impl Iterator<Item = &Transaction> {
        self.txs.iter().filter_map(|(_, tx)| match tx {
            TxNode::Whole(tx) => Some(tx),
            TxNode::Partial(_) => None,
        })
    }

    pub fn partial_transactions(&self) -> impl Iterator<Item = (Txid, &BTreeMap<u32, TxOut>)> {
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

    /// Previews the resultant [`Additions`] when [`Self`] is updated against the `update` graph.
    pub fn determine_additions(&self, update: &Self) -> Result<Additions, TxOutConflict> {
        let mut additions = Additions::default();

        for (&txid, update_tx) in &update.txs {
            match update_tx {
                TxNode::Whole(tx) => {
                    if matches!(self.txs.get(&txid), None | Some(TxNode::Partial(_))) {
                        additions.tx.insert(tx.clone());
                    }
                }
                TxNode::Partial(partial) => {
                    for (&vout, update_txout) in partial {
                        let outpoint = OutPoint::new(txid, vout);

                        let insert = match self.txouts(txid) {
                            Some(txouts) => match txouts.get(&vout) {
                                Some(&original_txout) if original_txout != update_txout => {
                                    return Err(TxOutConflict {
                                        outpoint,
                                        original_txout: original_txout.clone(),
                                        update_txout: update_txout.clone(),
                                    });
                                }
                                Some(_) => false,
                                None => true,
                            },
                            None => true,
                        };

                        if insert {
                            additions
                                .txout
                                .insert(OutPoint { txid, vout }, update_txout.clone());
                        }
                    }
                }
            }
        }

        Ok(additions)
    }

    /// Extends this graph with another so that `self` becomes the union of the two sets of
    /// transactions.
    pub fn apply_update(&mut self, update: TxGraph) -> Result<Additions, TxOutConflict> {
        let additions = self.determine_additions(&update)?;
        self.apply_additions(additions.clone());
        Ok(additions)
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
                TxNode::Whole(_) => {
                    debug_assert!(
                        false,
                        "graph additions must not replace whole tx with partial"
                    );
                }
                TxNode::Partial(txouts) => {
                    debug_assert!(
                        !matches!(txouts.get(&outpoint.vout), Some(original_txout) if original_txout != &txout),
                        "txout addition should not have different txout of same outpoint"
                    );
                    txouts.insert(outpoint.vout, txout);
                }
            }
        }
    }
}

/// Failure case that occurs when a [`TxOut`] is inserted at a given [`OutPoint`] in which a
/// non-matching [`TxOut`] already exists.
#[derive(Debug, Clone, PartialEq)]
pub struct TxOutConflict {
    pub outpoint: OutPoint,
    pub original_txout: TxOut,
    pub update_txout: TxOut,
}

impl core::fmt::Display for TxOutConflict {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cannot insert txout ({:?}) at outpoint ({}) as this replaces a non-matching txout ({:?})",
            self.update_txout, self.outpoint, self.original_txout)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TxOutConflict {}

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
