use bitcoin::{OutPoint, TxOut};

use crate::{collections::*, SparseChain, TxGraph, TxHeight};

#[derive(Clone, Debug, Default)]
pub struct UnspentIndex {
    utxos: HashMap<OutPoint, (TxOut, TxHeight)>,
    outpoints_by_height: BTreeSet<(TxHeight, OutPoint)>,
}

impl UnspentIndex {
    pub fn sync(&mut self, chain: &SparseChain, graph: &TxGraph) {
        let utxos = chain
            .iter_txids()
            .flat_map(|(h, txid)| {
                let tx = graph.tx(&txid).expect("tx of txid should exist");
                debug_assert_eq!(tx.txid(), txid);

                let height = TxHeight::from(h);

                tx.output.iter().enumerate().filter_map(move |(vout, txout)| {
                    let outpoint = OutPoint {
                            txid,
                            vout: vout as u32,
                        };

                    let is_unspent = graph
                        .is_unspent(&outpoint)
                        .expect("outpoint should exist");

                    if is_unspent {
                        Some((outpoint, (txout.clone(), height)))
                    } else {
                        None
                    }
                })
            })
            .collect::<HashMap<_, _>>();

        let utxos_by_height = utxos.iter().map(|(op, (_, h))| (*h, *op)).collect::<BTreeSet<_>>();

        self.utxos = utxos;
        self.outpoints_by_height = utxos_by_height;
    }

    // pub fn 
}
