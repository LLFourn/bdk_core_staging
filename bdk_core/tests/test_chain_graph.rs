#[macro_use]
mod common;

use bdk_core::{
    chain_graph::{ChainGraph, ChangeSet, UpdateFailure},
    sparse_chain,
    tx_graph::Additions,
    BlockId, TxHeight,
};
use bitcoin::{OutPoint, PackedLockTime, Script, Sequence, Transaction, TxIn, TxOut, Witness};

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
    cg1.insert_tx(tx1, Some(TxHeight::Unconfirmed)).unwrap();
    let mut cg2 = cg1.clone();
    cg1.insert_tx(tx2.clone(), Some(TxHeight::Unconfirmed))
        .unwrap();
    cg2.insert_tx(tx3.clone(), Some(TxHeight::Unconfirmed))
        .unwrap();
    // put the these txs in the graph but not in chain. `spent_by` should return the one that was
    // actually in the respective chain.
    cg1.insert_tx(tx3.clone(), None).expect("should insert");
    cg2.insert_tx(tx2.clone(), None).expect("should insert");

    assert_eq!(cg1.spent_by(op), Some((&TxHeight::Unconfirmed, tx2.txid())));
    assert_eq!(cg2.spent_by(op), Some((&TxHeight::Unconfirmed, tx3.txid())));
}

#[test]
fn update_evicts_conflicting_tx() {
    let cp_a = BlockId {
        height: 0,
        hash: h!("A"),
    };
    let cp_b = BlockId {
        height: 1,
        hash: h!("B"),
    };

    let tx_a = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };

    let tx_b = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint::new(tx_a.txid(), 0),
            script_sig: Script::new(),
            sequence: Sequence::default(),
            witness: Witness::new(),
        }],
        output: vec![TxOut::default()],
    };

    let tx_b2 = Transaction {
        version: 0x02,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint::new(tx_a.txid(), 0),
            script_sig: Script::new(),
            sequence: Sequence::default(),
            witness: Witness::new(),
        }],
        output: vec![TxOut::default(), TxOut::default()],
    };

    let cg1 = {
        let mut cg = ChainGraph::default();
        cg.insert_checkpoint(cp_a).expect("should insert cp");
        cg.insert_tx(tx_a.clone(), Some(TxHeight::Confirmed(0)))
            .expect("should insert tx");
        cg.insert_tx(tx_b.clone(), Some(TxHeight::Unconfirmed))
            .expect("should insert tx");
        cg
    };
    let cg2 = {
        let mut cg = ChainGraph::default();
        cg.insert_tx(tx_b2.clone(), Some(TxHeight::Unconfirmed))
            .expect("should insert tx");
        cg
    };
    assert_eq!(
        cg1.determine_changeset(&cg2),
        Ok(ChangeSet::<TxHeight> {
            chain: sparse_chain::ChangeSet {
                checkpoints: Default::default(),
                txids: [
                    (tx_b.txid(), None),
                    (tx_b2.txid(), Some(TxHeight::Unconfirmed))
                ]
                .into()
            },
            graph: Additions {
                tx: [tx_b2.clone()].into(),
                txout: [].into()
            },
        }),
        "tx should be evicted from mempool"
    );

    let cg1 = {
        let mut cg = ChainGraph::default();
        cg.insert_checkpoint(cp_a).expect("should insert cp");
        cg.insert_checkpoint(cp_b).expect("should insert cp");
        cg.insert_tx(tx_a.clone(), Some(TxHeight::Confirmed(0)))
            .expect("should insert tx");
        cg.insert_tx(tx_b.clone(), Some(TxHeight::Confirmed(1)))
            .expect("should insert tx");
        cg
    };
    let cg2 = {
        let mut cg = ChainGraph::default();
        cg.insert_tx(tx_b2.clone(), Some(TxHeight::Unconfirmed))
            .expect("should insert tx");
        cg
    };
    assert_eq!(
        cg1.determine_changeset(&cg2),
        Err(UpdateFailure::Conflict {
            already_confirmed_tx: (TxHeight::Confirmed(1), tx_b.txid()),
            update_tx: (TxHeight::Unconfirmed, tx_b2.txid()),
        }),
        "fail if tx is evicted from valid block"
    );
}
