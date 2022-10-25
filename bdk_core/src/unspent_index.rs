use core::ops::{Bound, RangeBounds};

use bitcoin::{hashes::Hash, OutPoint, TxOut, Txid};

use crate::{collections::*, ChangeSet, SpkTracker, SyncFailure, TxHeight, Vec};

#[derive(Clone, Debug)]
pub struct UnspentIndex<I> {
    /// Map of outpoints to (txout, confirmation height)
    utxos: HashMap<OutPoint, (I, TxOut, TxHeight)>,
    /// Outpoints ordered by confirmation height
    ordered_outpoints: BTreeSet<(TxHeight, OutPoint)>,
}

#[derive(Clone, Debug)]
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
        spk_tracker: &SpkTracker<I>,
        changes: &ChangeSet,
    ) -> Result<(), SyncFailure> {
        let txo_changes = changes
            .txids
            .iter()
            .flat_map(|(txid, h_delta)| -> Vec<_> {
                let txouts = spk_tracker.range_tx_outputs(*txid);

                // ensure `height_change.from` is consistent with index state
                match h_delta.from {
                    // outpoints should all exist
                    Some(exp_height) => txouts
                        .map(|(spk_i, op, txout)| -> Result<_, SyncFailure> {
                            let (_, index_txout, height) = self
                                .utxos
                                .get(&op)
                                .ok_or_else(|| SyncFailure::TxNotInIndex(*txid))?;
                            debug_assert_eq!(index_txout, txout);

                            if height != &exp_height {
                                Err(SyncFailure::TxInconsistent {
                                    txid: *txid,
                                    original: Some(*height),
                                    change: h_delta.clone(),
                                })
                            } else {
                                Ok((spk_i, op, txout, h_delta))
                            }
                        })
                        .collect::<Vec<_>>(),
                    // outpoints should all not exist
                    None => txouts
                        .map(|(spk_i, op, txout)| -> Result<_, SyncFailure> {
                            if let Some((_, _, tx_height)) = self.utxos.get(&op) {
                                Err(SyncFailure::TxInconsistent {
                                    txid: *txid,
                                    original: Some(*tx_height),
                                    change: h_delta.clone(),
                                })
                            } else {
                                Ok((spk_i, op, txout, h_delta))
                            }
                        })
                        .collect::<Vec<_>>(),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // apply changes
        for (spk_i, op, txout, height_change) in txo_changes {
            if let Some(from_height) = height_change.from {
                self.utxos.remove(&op);
                self.ordered_outpoints.remove(&(from_height, op));
            }

            match height_change.to {
                Some(new_height) => {
                    self.utxos.insert(op, (spk_i, txout.clone(), new_height));
                    self.ordered_outpoints.insert((new_height, op));
                }
                None => {
                    if let Some((_, _, h)) = self.utxos.remove(&op) {
                        self.ordered_outpoints.remove(&(h, op));
                    }
                }
            }
        }

        Ok(())
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
