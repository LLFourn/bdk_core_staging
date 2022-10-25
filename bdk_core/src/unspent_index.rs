use core::ops::{Bound, RangeBounds};

use bitcoin::{hashes::Hash, OutPoint, TxOut, Txid};

use crate::{collections::*, ChangeSet, SyncFailure, TxGraph, TxHeight, Vec};

#[derive(Clone, Debug, Default)]
pub struct UnspentIndex {
    /// Map of outpoints to (txout, confirmation height)
    utxos: HashMap<OutPoint, (TxOut, TxHeight)>,
    /// Outpoints ordered by confirmation height
    ordered_outpoints: BTreeSet<(TxHeight, OutPoint)>,
}

#[derive(Clone, Debug)]
pub struct Unspent {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub height: TxHeight,
}

impl UnspentIndex {
    /// Given a [`ChangeSet`] and a [`TxGraph`], we sync the [`UnspentIndex`].
    ///
    /// TODO: Figure out how to make this cleaner and more efficient.
    pub fn sync(&mut self, graph: &TxGraph, changes: &ChangeSet) -> Result<(), SyncFailure> {
        let txo_changes = changes
            .txids
            .iter()
            .flat_map(|(txid, h_delta)| -> Vec<_> {
                // obtain iterator over (outpoint, txouts) of given txid
                let txouts = match graph.tx(*txid) {
                    Some(tx) => tx
                        .output
                        .iter()
                        .enumerate()
                        .map(|(vout, txout)| (OutPoint::new(*txid, vout as _), txout, h_delta)),
                    None => return vec![Err(SyncFailure::TxNotInGraph(*txid))],
                };

                // ensure `height_change.from` is consistent with index state
                match h_delta.from {
                    // outpoints should all exist
                    Some(exp_height) => txouts
                        .map(|(op, txout, h_delta)| -> Result<_, SyncFailure> {
                            let (index_txout, height) = self
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
                                Ok((op, txout, h_delta))
                            }
                        })
                        .collect::<Vec<_>>(),
                    // outpoints should all not exist
                    None => txouts
                        .map(|(op, txout, h_delta)| -> Result<_, SyncFailure> {
                            if let Some((_, tx_height)) = self.utxos.get(&op) {
                                Err(SyncFailure::TxInconsistent {
                                    txid: *txid,
                                    original: Some(*tx_height),
                                    change: h_delta.clone(),
                                })
                            } else {
                                Ok((op, txout, h_delta))
                            }
                        })
                        .collect::<Vec<_>>(),
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // apply changes
        for (op, txout, height_change) in txo_changes {
            if let Some(from_height) = height_change.from {
                self.utxos.remove(&op);
                self.ordered_outpoints.remove(&(from_height, op));
            }

            match height_change.to {
                Some(new_height) => {
                    self.utxos.insert(op, (txout.clone(), new_height));
                    self.ordered_outpoints.insert((new_height, op));
                }
                None => {
                    if let Some((_, h)) = self.utxos.remove(&op) {
                        self.ordered_outpoints.remove(&(h, op));
                    }
                }
            }
        }

        Ok(())
    }

    /// Obtain [`Unspent`] from given [`OutPoint`].
    pub fn unspent(&self, outpoint: OutPoint) -> Option<Unspent> {
        self.utxos.get(&outpoint).map(|(txout, height)| Unspent {
            outpoint,
            txout: txout.clone(),
            height: *height,
        })
    }

    /// Iterate all unspent outputs (UTXOs), from most confirmations to least confirmations.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = Unspent> + ExactSizeIterator + '_ {
        self.ordered_outpoints
            .iter()
            .map(|&(height, outpoint)| Unspent {
                outpoint,
                txout: self.utxos[&outpoint].0.clone(),
                height,
            })
    }

    /// Range unspent outputs (UTXO) by confirmation height ([`TxHeight`]).
    pub fn range<R>(&self, range: R) -> impl DoubleEndedIterator<Item = Unspent> + '_
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
            .map(|&(height, outpoint)| Unspent {
                outpoint,
                txout: self.utxos[&outpoint].0.clone(),
                height,
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
