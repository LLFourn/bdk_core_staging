#[macro_use]
mod common;
use bdk_chain::{
    collections::*,
    tx_graph::{Additions, TxGraph},
};
use bitcoin::{hashes::Hash, OutPoint, PackedLockTime, Script, Transaction, TxIn, TxOut, Txid};
use core::iter;

#[test]
fn insert_txouts() {
    let original_ops = [
        (
            OutPoint::new(h!("tx1"), 1),
            TxOut {
                value: 10_000,
                script_pubkey: Script::new(),
            },
        ),
        (
            OutPoint::new(h!("tx1"), 2),
            TxOut {
                value: 20_000,
                script_pubkey: Script::new(),
            },
        ),
    ];

    let update_ops = [(
        OutPoint::new(h!("tx2"), 0),
        TxOut {
            value: 20_000,
            script_pubkey: Script::new(),
        },
    )];

    let mut graph = {
        let mut graph = TxGraph::default();
        for (outpoint, txout) in &original_ops {
            assert_eq!(
                graph.insert_txout(*outpoint, txout.clone()),
                Additions {
                    txout: [(*outpoint, txout.clone())].into(),
                    ..Default::default()
                }
            );
        }
        graph
    };

    let update = {
        let mut graph = TxGraph::default();
        for (outpoint, txout) in &update_ops {
            assert_eq!(
                graph.insert_txout(*outpoint, txout.clone()),
                Additions {
                    txout: [(*outpoint, txout.clone())].into(),
                    ..Default::default()
                }
            );
        }
        graph
    };

    let additions = graph.determine_additions(&update);

    assert_eq!(
        additions,
        Additions {
            tx: [].into(),
            txout: update_ops.into(),
        }
    );

    graph.apply_additions(additions);
    assert_eq!(graph.all_txouts().count(), 3);
    assert_eq!(graph.full_transactions().count(), 0);
    assert_eq!(graph.partial_transactions().count(), 2);
}

#[test]
fn insert_tx_graph_doesnt_count_coinbase_as_spent() {
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            ..Default::default()
        }],
        output: vec![],
    };

    let mut graph = TxGraph::default();
    let _ = graph.insert_tx(tx);
    assert!(graph.outspends(OutPoint::null()).is_empty());
    assert!(graph.tx_outspends(Txid::all_zeros()).next().is_none());
}

#[test]
fn insert_tx_graph_keeps_track_of_spend() {
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

    let mut graph1 = TxGraph::default();
    let mut graph2 = TxGraph::default();

    // insert in different order
    let _ = graph1.insert_tx(tx1.clone());
    let _ = graph1.insert_tx(tx2.clone());

    let _ = graph2.insert_tx(tx2.clone());
    let _ = graph2.insert_tx(tx1.clone());

    assert_eq!(
        &*graph1.outspends(op),
        &iter::once(tx2.txid()).collect::<HashSet<_>>()
    );
    assert_eq!(graph2.outspends(op), graph1.outspends(op));
}

#[test]
fn insert_tx_can_retrieve_full_tx_from_graph() {
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            ..Default::default()
        }],
        output: vec![TxOut::default()],
    };

    let mut graph = TxGraph::default();
    let _ = graph.insert_tx(tx.clone());
    assert_eq!(graph.get_tx(tx.txid()), Some(&tx));
}

#[test]
fn insert_tx_displaces_txouts() {
    let mut tx_graph = TxGraph::default();
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut {
            value: 42_000,
            script_pubkey: Script::default(),
        }],
    };

    let _ = tx_graph.insert_txout(
        OutPoint {
            txid: tx.txid(),
            vout: 0,
        },
        TxOut {
            value: 1337_000,
            script_pubkey: Script::default(),
        },
    );

    let _ = tx_graph.insert_txout(
        OutPoint {
            txid: tx.txid(),
            vout: 0,
        },
        TxOut {
            value: 1_000_000_000,
            script_pubkey: Script::default(),
        },
    );

    let _additions = tx_graph.insert_tx(tx.clone());

    assert_eq!(
        tx_graph
            .get_txout(OutPoint {
                txid: tx.txid(),
                vout: 0
            })
            .unwrap()
            .value,
        42_000
    );
    assert_eq!(
        tx_graph.get_txout(OutPoint {
            txid: tx.txid(),
            vout: 1
        }),
        None
    );
}

#[test]
fn insert_txout_does_not_displace_tx() {
    let mut tx_graph = TxGraph::default();
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut {
            value: 42_000,
            script_pubkey: Script::default(),
        }],
    };

    let _additions = tx_graph.insert_tx(tx.clone());

    let _ = tx_graph.insert_txout(
        OutPoint {
            txid: tx.txid(),
            vout: 0,
        },
        TxOut {
            value: 1337_000,
            script_pubkey: Script::default(),
        },
    );

    let _ = tx_graph.insert_txout(
        OutPoint {
            txid: tx.txid(),
            vout: 0,
        },
        TxOut {
            value: 1_000_000_000,
            script_pubkey: Script::default(),
        },
    );

    assert_eq!(
        tx_graph
            .get_txout(OutPoint {
                txid: tx.txid(),
                vout: 0
            })
            .unwrap()
            .value,
        42_000
    );
    assert_eq!(
        tx_graph.get_txout(OutPoint {
            txid: tx.txid(),
            vout: 1
        }),
        None
    );
}

#[test]
fn test_calculate_fee() {
    let mut graph = TxGraph::default();
    let intx1 = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut {
            value: 100,
            ..Default::default()
        }],
    };
    let intx2 = Transaction {
        version: 0x02,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut {
            value: 200,
            ..Default::default()
        }],
    };

    let intxout1 = (
        OutPoint {
            txid: h!("dangling output"),
            vout: 0,
        },
        TxOut {
            value: 300,
            ..Default::default()
        },
    );

    let _ = graph.insert_tx(intx1.clone());
    let _ = graph.insert_tx(intx2.clone());
    let _ = graph.insert_txout(intxout1.0, intxout1.1);

    let mut tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![
            TxIn {
                previous_output: OutPoint {
                    txid: intx1.txid(),
                    vout: 0,
                },
                ..Default::default()
            },
            TxIn {
                previous_output: OutPoint {
                    txid: intx2.txid(),
                    vout: 0,
                },
                ..Default::default()
            },
            TxIn {
                previous_output: intxout1.0,
                ..Default::default()
            },
        ],
        output: vec![TxOut {
            value: 500,
            ..Default::default()
        }],
    };

    assert_eq!(graph.calculate_fee(&tx), Some(100));

    tx.input.remove(2);

    // fee would be negative
    assert_eq!(graph.calculate_fee(&tx), Some(-200));

    // If we have an unknown outpoint, fee should return None.
    tx.input.push(TxIn {
        previous_output: OutPoint {
            txid: h!("unknown_txid"),
            vout: 0,
        },
        ..Default::default()
    });
    assert_eq!(graph.calculate_fee(&tx), None);
}

#[test]
fn test_calculate_fee_on_coinbase() {
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint::null(),
            ..Default::default()
        }],
        output: vec![TxOut::default()],
    };

    let graph = TxGraph::default();

    assert_eq!(graph.calculate_fee(&tx), Some(0));
}
