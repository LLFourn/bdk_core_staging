#[macro_use]
mod common;
use bdk_core::{
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
            assert!(graph.insert_txout(*outpoint, txout.clone()));
        }
        graph
    };

    let update = {
        let mut graph = TxGraph::default();
        for (outpoint, txout) in &update_ops {
            assert!(graph.insert_txout(*outpoint, txout.clone()));
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
    assert_eq!(graph.iter_all_txouts().count(), 3);
    assert_eq!(graph.iter_full_transactions().count(), 0);
    assert_eq!(graph.iter_partial_transactions().count(), 2);
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
    graph.insert_tx(tx);
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
    graph1.insert_tx(tx1.clone());
    graph1.insert_tx(tx2.clone());

    graph2.insert_tx(tx2.clone());
    graph2.insert_tx(tx1.clone());

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
    graph.insert_tx(tx.clone());
    assert_eq!(graph.tx(tx.txid()), Some(&tx));
}
