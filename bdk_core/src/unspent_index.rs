use core::ops::{Bound, RangeBounds};

use bitcoin::{hashes::Hash, OutPoint, TxOut, Txid};

use crate::{collections::*, SparseChain, TxGraph, TxHeight};

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
    pub fn sync(&mut self, chain: &SparseChain, graph: &TxGraph) {
        let utxos = chain
            .iter_txids()
            .flat_map(|(h, txid)| {
                let tx = graph.tx(&txid).expect("tx of txid should exist");
                debug_assert_eq!(tx.txid(), txid);

                let height = TxHeight::from(h);

                tx.output
                    .iter()
                    .enumerate()
                    .filter_map(move |(vout, txout)| {
                        let outpoint = OutPoint {
                            txid,
                            vout: vout as u32,
                        };

                        let is_unspent =
                            graph.is_unspent(&outpoint).expect("outpoint should exist");

                        if is_unspent {
                            Some((outpoint, (txout.clone(), height)))
                        } else {
                            None
                        }
                    })
            })
            .collect::<HashMap<_, _>>();

        let utxos_by_height = utxos
            .iter()
            .map(|(op, (_, h))| (*h, *op))
            .collect::<BTreeSet<_>>();

        self.utxos = utxos;
        self.ordered_outpoints = utxos_by_height;
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
