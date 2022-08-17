use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt::{Debug, Display},
    ops::RangeBounds,
};

use bitcoin::{OutPoint, Transaction, Txid};

use super::*;

/// Represents an error in bdk core.
#[derive(Debug)]
pub enum CoreError {
    /// Generic error.
    Generic(&'static str),
    /// We should check for reorg.
    ReorgDetected,
}

impl Display for CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "core error: {:?}", self)
    }
}

impl std::error::Error for CoreError {}

/// Block header data that we are interested in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialHeader {
    /// Block hash
    pub hash: BlockHash,
    /// Block time
    pub time: u32,
}

/// Represents a candidate transaction to be introduced to [SparseChain].
pub struct CandidateTx {
    /// Txid of candidate.
    pub txid: Txid,
    /// Confirmed height and header (if any).
    pub confirmed_at: Option<(u32, PartialHeader)>,
}

/// This is a transactions with `confirmed_at`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainTx {
    /// The raw transaction.
    pub tx: Transaction,
    /// Confirmed at (if any).
    pub confirmed_at: Option<BlockTime>,
}

/// Saves a [SparseChain] in persistent storage.
pub trait SparseChainPersister {
    /// Write to persistent storage.
    ///
    /// * `from_block`: write blocks from block of this index
    /// * `from_tx`: write transactions from transaction of this index
    ///
    /// Everything after `from_block` and `from_tx` is to be cleared and replaced with `delta`.
    fn rewrite_from<'a, B, T>(
        &self,
        from_block: usize,
        from_tx: usize,
        block_iter: B,
        tx_iter: T,
    ) -> Result<(), CoreError>
    where
        B: Iterator<Item = (&'a u32, &'a PartialHeader)>,
        T: Iterator<Item = (&'a (u32, Txid), &'a Transaction)>;
}

/// SparseChain implementation
pub struct AlternativeSparseChain {
    // relevant blocks which contain transactions we are interested in: <height: (block_hash, block_time)>
    pub(crate) blocks: BTreeMap<u32, PartialHeader>,
    // relevant transactions lexicographically ordered by (block_height, txid)
    // unconfirmed txs have block_height as `u32::Max`
    pub(crate) txs: BTreeMap<(u32, Txid), Transaction>,
    // last attempted spends that we are aware of <spent_outpoint, (spending_txid, spending_vin)>
    // TODO: Do we need to record multiple spends?
    pub(crate) spends: BTreeMap<OutPoint, Txid>,
    // ref from tx to block height
    pub(crate) at_height: HashMap<Txid, u32>,
    // records persistence storage state changes (None: no change, Some: changes from...)
    pub(crate) persist_from: Option<(u32, Txid)>,
}

impl AlternativeSparseChain {
    /// Iterates all txs.
    pub fn iter_txs(&self) -> impl DoubleEndedIterator<Item = ChainTx> + '_ {
        self.txs.iter().map(move |((height, _), tx)| ChainTx {
            tx: tx.clone(),
            confirmed_at: self.get_confirmed_at(height),
        })
    }

    /// Iterates [PartialHeader]s of relevant blocks.
    pub fn iter_blocks(
        &self,
        range: impl RangeBounds<u32>,
    ) -> impl DoubleEndedIterator<Item = (&u32, &PartialHeader)> + '_ {
        self.blocks.range(range)
    }

    /// Returns whether an output of [OutPoint] is spent.
    pub fn is_spent(&self, outpoint: &OutPoint) -> bool {
        self.outspend(outpoint).is_some()
    }

    /// If the output is spent by a transaction that is tracked, this returns the [Txid] that spent
    /// it. Otherwise, we return [None].
    pub fn outspend(&self, outpoint: &OutPoint) -> Option<Txid> {
        self.spends
            .get(outpoint)
            .and_then(|txid| {
                if self.at_height.contains_key(txid) {
                    Some(txid)
                } else {
                    None
                }
            })
            .cloned()
    }

    /// The outputs from the transaction with id `txid` that have been spent.
    ///
    /// Each item contains the output index and the txid that spent that output.
    pub fn outspends(&self, txid: Txid) -> impl DoubleEndedIterator<Item = (u32, Txid)> + '_ {
        let start = OutPoint { txid, vout: 0 };
        let end = OutPoint {
            txid,
            vout: u32::MAX,
        };
        self.spends
            .range(start..=end)
            .filter(move |(_, txid)| self.at_height.contains_key(*txid))
            .map(|(outpoint, txid)| (outpoint.vout, *txid))
    }

    /// Obtain a [FullTxOut].
    pub fn full_txout(&self, outpoint: &OutPoint) -> Option<FullTxOut> {
        let height = *self.at_height.get(&outpoint.txid)?;
        let tx = self
            .txs
            .get(&(height, outpoint.txid))
            .expect("a tx back-ref should always be associated with an actual tx");

        let txout = tx.output.get(outpoint.vout as usize)?;
        let confirmed_at = self.get_confirmed_at(&height);
        let spent_by = self.outspend(outpoint);

        Some(FullTxOut {
            outpoint: *outpoint,
            confirmed_at,
            spent_by,
            value: txout.value,
            script_pubkey: txout.script_pubkey.clone(),
        })
    }

    /// Given the introduced transactions, calculate the deltas to be applied to [SparseChain].
    ///
    /// TODO: Conflict detection.
    pub fn calculate_deltas<I>(&self, mut candidates: I) -> Result<Delta<Unfilled>, CoreError>
    where
        I: Iterator<Item = CandidateTx>,
    {
        // candidate changes
        let mut deltas = Delta::default();

        candidates.try_for_each(
            |CandidateTx { txid, confirmed_at }| -> Result<(), CoreError> {
                // unconfirmed transactions are internally stored with height `u32::MAX`
                let tx_height = confirmed_at.map(|(h, _)| h).unwrap_or(u32::MAX);
                let tx_key = (tx_height, txid);

                // if tx of (height, txid) already exists, skip
                if deltas.tx_keys.contains(&tx_key) || self.txs.contains_key(&tx_key) {
                    println!(
                        "tx {} at height {} already exists, skipping",
                        txid, tx_height
                    );
                    return Ok(());
                }

                // if txid moved height, and the original height is not u32::MAX (unconfirmed),
                // report reorg
                if matches!(self.at_height.get(&txid), Some(h) if *h != u32::MAX) {
                    return Err(CoreError::ReorgDetected);
                }

                // if candidate tx is confirmed, check that the candidate block does not conflict with
                // blocks we know of
                if let Some((h, candidate_header)) = &confirmed_at {
                    debug_assert_eq!(tx_height, *h);

                    match deltas.blocks.get(h).or_else(|| self.blocks.get(h)) {
                        Some(header) => {
                            // expect candidate block to be the same as existing block of same height,
                            // otherwise we have a reorg
                            if header != candidate_header {
                                return Err(CoreError::ReorgDetected);
                            }
                        }
                        None => {
                            // no block exists at height, introduce candidate block
                            deltas.blocks.insert(*h, *candidate_header);
                        }
                    };
                }

                Ok(())
            },
        )?;

        Ok(deltas)
    }

    /// Apply [Delta] to [SparseChain].
    pub fn apply_delta(&mut self, delta: Delta<Filled>) -> Result<(), CoreError> {
        delta.apply_to_sparsechain(self)
    }

    /// Flush [SparseChain] changes into persistence storage.
    pub fn flush<P: SparseChainPersister>(&mut self, p: P) -> Result<bool, CoreError> {
        match self.persist_from {
            Some((height, txid)) => {
                let from_block = self.blocks.range(..height).count();
                let from_tx = self.txs.range(..(height, txid)).count();
                let block_iter = self.blocks.range(height..);
                let tx_iter = self.txs.range((height, txid)..);

                p.rewrite_from(from_block, from_tx, block_iter, tx_iter)?;
                self.persist_from = None;
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Rollback all transactions from the given height and above and return the resultant [Delta].
    ///
    /// WARNING: The resultant [Delta<Negated>] should be applied to all `SpkTracker`s associated
    /// with this [SparseChain], otherwise the `SpkTracker`s will end up in an inconsistent state.
    pub fn rollback(&mut self, height: u32) -> Delta<Negated> {
        let key = (height, Txid::default());

        let removed_blocks = self.blocks.split_off(&height);
        let removed_txs = self.txs.split_off(&key);

        let mut delta = Delta::<Negated> {
            blocks: removed_blocks,
            tx_keys: BTreeSet::new(),
            tx_values: HashMap::with_capacity(removed_txs.len()),
            ..Default::default()
        };

        for ((height, txid), tx) in removed_txs {
            // remove back ref
            assert_eq!(self.at_height.remove(&txid), Some(height));

            delta.tx_keys.insert((height, txid));
            delta.tx_values.insert(txid, tx);
        }

        self.update_persist_from(key);

        delta
    }

    /// Clear all unconfirmed txs, returning the resultant delta.
    ///
    /// This is the same as calling [SparseChain]::rollback(u32::MAX).
    pub fn remove_unconfirmed(&mut self) -> Delta<Negated> {
        self.rollback(u32::MAX)
    }

    /// Selectively remove multiple transactions of txids.
    pub fn remove_txs<I: Iterator<Item = Txid>>(&mut self, txids: I) -> Delta<Negated> {
        let mut delta = Delta {
            tx_values: HashMap::with_capacity({
                let (lower, upper) = txids.size_hint();
                upper.unwrap_or(lower)
            }),
            ..Default::default()
        };

        for txid in txids {
            let removed_tx = self.at_height.remove_entry(&txid).map(|(txid, height)| {
                self.txs
                    .remove_entry(&(height, txid))
                    .expect("a previous operation forgot to clear the tx-back-ref in `::at_height`")
            });

            if let Some(((height, txid), tx)) = removed_tx {
                self.update_persist_from((height, txid));

                delta.tx_keys.insert((height, txid));
                delta.tx_values.insert(txid, tx);
            }
        }

        // clear empty blocks
        if let Some((from_height, _)) = self.persist_from {
            delta.blocks = self.remove_irrelevant_blocks(from_height);
        }

        delta
    }

    /// Get transaction of txid.
    pub fn get_tx(&self, txid: Txid) -> Option<ChainTx> {
        self.at_height.get(&txid).map(|&height| {
            let tx = self
                .txs
                .get(&(height, txid))
                .expect("tx back ref was not cleared")
                .clone();

            ChainTx {
                tx,
                confirmed_at: self.get_confirmed_at(&height),
            }
        })
    }

    /// helper: get [BlockTime] from `height`
    fn get_confirmed_at(&self, height: &u32) -> Option<BlockTime> {
        let at = self
            .blocks
            .get(height)
            .map(|PartialHeader { time, .. }| BlockTime {
                height: *height,
                time: *time as u64,
            });

        assert_eq!(
            *height == u32::MAX,
            at.is_none(),
            "when height is MAX, it should represent an unconfirmed tx"
        );

        at
    }

    /// helper: clear irrelevant blocks from `persist_from`
    ///
    /// irrelevant blocks are blocks which do not have transactions that we track
    fn remove_irrelevant_blocks(&mut self, from_height: u32) -> BTreeMap<u32, PartialHeader> {
        let irrelevant_heights = self
            .blocks
            .range(from_height..)
            .filter_map(|(&height, _)| {
                // get count of txs of height
                let tx_count = self
                    .txs
                    .range((height, Txid::default())..(height + 1, Txid::default()))
                    .count();

                // mark for deletion if height has no txs
                if tx_count == 0 {
                    Some(height)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        irrelevant_heights
            .iter()
            .filter_map(|height| self.blocks.remove_entry(height))
            .collect()
    }

    /// helper: update `persist_from`
    fn update_persist_from(&mut self, from: (u32, Txid)) {
        self.persist_from = Some(match self.persist_from {
            Some(persist_from) => std::cmp::min(persist_from, from),
            None => from,
        });
    }
}
