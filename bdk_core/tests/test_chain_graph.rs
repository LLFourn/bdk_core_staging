use bdk_core::{chain_graph::ChainGraph, TxHeight};
use bitcoin::{OutPoint, PackedLockTime, Transaction, TxIn, TxOut};

#[test]
fn test_spent_by() {
    let tx1 = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };

    let op = OutPoint {
        txid: tx1.txid(),
        vout: 0,
    };

    let tx2 = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: op,
            ..Default::default()
        }],
        output: vec![],
    };
    let tx3 = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(42),
        input: vec![TxIn {
            previous_output: op,
            ..Default::default()
        }],
        output: vec![],
    };

    let mut cg1 = ChainGraph::default();
    cg1.insert_tx(tx1, TxHeight::Unconfirmed).unwrap();
    let mut cg2 = cg1.clone();
    cg1.insert_tx(tx2.clone(), TxHeight::Unconfirmed).unwrap();
    cg2.insert_tx(tx3.clone(), TxHeight::Unconfirmed).unwrap();
    // put the these txs in the graph but not in chain. `spent_by` should return the one that was
    // actually in the respective chain.
    cg1.graph.insert_tx(tx3.clone());
    cg2.graph.insert_tx(tx2.clone());

    assert_eq!(cg1.spent_by(op), Some((&TxHeight::Unconfirmed, tx2.txid())));
    assert_eq!(cg2.spent_by(op), Some((&TxHeight::Unconfirmed, tx3.txid())));
}
