use core::ops::{Bound, RangeBounds};

use bitcoin::{hashes::Hash, OutPoint, TxOut, Txid};

use crate::{collections::*, SparseChain, SpkTracker, SyncFailure, TxGraph, TxHeight};

#[derive(Clone, Debug)]
pub struct UnspentIndex<I> {
    /// Map of outpoints to (txout, confirmation height)
    utxos: HashMap<OutPoint, (I, TxOut, TxHeight)>,
    /// Outpoints ordered by confirmation height
    ordered_outpoints: BTreeSet<(TxHeight, OutPoint)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unspent<I> {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub spk_index: I,
    pub height: TxHeight,
}

impl<I: Clone + Ord> Default for UnspentIndex<I> {
    fn default() -> Self {
        Self {
            utxos: Default::default(),
            ordered_outpoints: Default::default(),
        }
    }
}

impl<I: Clone + Ord> UnspentIndex<I> {
    /// Given a [`ChangeSet`] and a [`TxGraph`], we sync the [`UnspentIndex`].
    ///
    /// TODO: Figure out how to make this cleaner and more efficient.
    pub fn sync(
        &mut self,
        chain: &SparseChain,
        graph: &TxGraph,
        tracker: &SpkTracker<I>,
    ) -> Result<(), SyncFailure> {
        // clear all
        self.utxos.clear();
        self.ordered_outpoints.clear();

        // txout must:
        // 1. exist in chain (also get height)
        // 2. not be spent by txid that still exists in chan
        tracker
            .iter_txout()
            .filter_map(|(spk_i, op, txo)| {
                // txout must exit in chain
                let h = chain.transaction_height(op.txid)?;

                // txout must not be spent by txid that still exists in chain
                if let Some(spends) = graph.outspend(&op) {
                    let is_spent = spends
                        .iter()
                        .find(|&&txid| chain.transaction_height(txid).is_some())
                        .is_some();
                    if is_spent {
                        return None;
                    }
                }

                Some((spk_i, op, txo, h))
            })
            .for_each(|(spk_i, op, txo, h)| {
                self.insert_unspent(spk_i, op, txo.clone(), h);
            });

        Ok(())
    }

    /// Inserts or replaces a single UTXO. Returns true when UTXO is changed or replaced.
    fn insert_unspent(
        &mut self,
        spk_index: I,
        outpoint: OutPoint,
        txout: TxOut,
        height: TxHeight,
    ) -> bool {
        let (_, _, new_height) = &*self
            .utxos
            .entry(outpoint)
            .or_insert((spk_index, txout, height));

        if *new_height != height {
            let removed = self.ordered_outpoints.remove(&(*new_height, outpoint));
            debug_assert!(removed, "inconsistent unspent index");
        }

        self.ordered_outpoints.insert((*new_height, outpoint))
    }

    /// Removes a single UTXO (if any).
    fn _remove_unspent(&mut self, outpoint: OutPoint) -> Option<(I, TxOut, TxHeight)> {
        let (spk_i, txout, height) = self.utxos.remove(&outpoint)?;
        let removed = self.ordered_outpoints.remove(&(height, outpoint));
        debug_assert!(removed, "inconsistent unspent_index fields");
        Some((spk_i, txout, height))
    }

    /// Obtain [`Unspent`] from given [`OutPoint`].
    pub fn unspent(&self, outpoint: OutPoint) -> Option<Unspent<I>> {
        self.utxos
            .get(&outpoint)
            .map(|(spk_index, txout, height)| Unspent {
                outpoint,
                txout: txout.clone(),
                spk_index: spk_index.clone(),
                height: *height,
            })
    }

    /// Iterate all unspent outputs (UTXOs), from most confirmations to least confirmations.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = Unspent<I>> + ExactSizeIterator + '_ {
        self.ordered_outpoints.iter().map(|&(height, outpoint)| {
            let (spk_i, txout, _) = &self.utxos[&outpoint];
            Unspent {
                outpoint,
                txout: txout.clone(),
                spk_index: spk_i.clone(),
                height,
            }
        })
    }

    /// Range unspent outputs (UTXO) by confirmation height ([`TxHeight`]).
    pub fn range<R>(&self, range: R) -> impl DoubleEndedIterator<Item = Unspent<I>> + '_
    where
        R: RangeBounds<TxHeight>,
    {
        let start = match range.start_bound() {
            Bound::Included(h) => Bound::Included((*h, min_outpoint())),
            Bound::Excluded(h) => Bound::Excluded((*h, max_outpoint())),
            Bound::Unbounded => Bound::Unbounded,
        };
        let end = match range.end_bound() {
            Bound::Included(h) => Bound::Included((*h, max_outpoint())),
            Bound::Excluded(h) => Bound::Excluded((*h, min_outpoint())),
            Bound::Unbounded => Bound::Unbounded,
        };
        self.ordered_outpoints
            .range((start, end))
            .map(|&(height, outpoint)| {
                let (spk_i, txout, _) = &self.utxos[&outpoint];
                Unspent {
                    outpoint,
                    txout: txout.clone(),
                    spk_index: spk_i.clone(),
                    height,
                }
            })
    }
}

fn min_outpoint() -> OutPoint {
    OutPoint {
        txid: Txid::from_slice(&[u8::MIN; 32]).unwrap(),
        vout: u32::MIN,
    }
}

fn max_outpoint() -> OutPoint {
    OutPoint {
        txid: Txid::from_slice(&[u8::MAX; 32]).unwrap(),
        vout: u32::MAX,
    }
}
