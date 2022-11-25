use bitcoin::{Block, OutPoint, Transaction, TxOut};

/// Trait to do something with every txout contained in a structure.
///
/// We would prefer just work with things that can give us a `Iterator<Item=TxOut>` here but rust's type
/// system makes it extremely hard to do this (without trait objects).
pub trait ForEachTxout {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut)));
}

impl ForEachTxout for Transaction {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        let txid = self.txid();
        for (i, txout) in self.output.iter().enumerate() {
            f((
                OutPoint {
                    txid,
                    vout: i as u32,
                },
                txout,
            ))
        }
    }
}

impl ForEachTxout for Block {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        for tx in self.txdata.iter() {
            tx.for_each_txout(f)
        }
    }
}
