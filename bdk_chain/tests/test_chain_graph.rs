#[macro_use]
mod common;

use bdk_chain::{
    chain_graph::{ChainGraph, ChangeSet, InflateFailure, UnresolvableConflict, UpdateFailure},
    collections::HashSet,
    sparse_chain,
    tx_graph::{self, Additions},
    BlockId, TxHeight,
};
use bitcoin::{
    OutPoint, PackedLockTime, Script, Sequence, Transaction, TxIn, TxOut, Txid, Witness,
};

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
    let _ = cg1
        .insert_tx(tx1, TxHeight::Unconfirmed)
        .expect("should insert");
    let mut cg2 = cg1.clone();
    let _ = cg1
        .insert_tx(tx2.clone(), TxHeight::Unconfirmed)
        .expect("should insert");
    let _ = cg2
        .insert_tx(tx3.clone(), TxHeight::Unconfirmed)
        .expect("should insert");

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
    let cp_b2 = BlockId {
        height: 1,
        hash: h!("B'"),
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
    {
        let mut cg1 = {
            let mut cg = ChainGraph::default();
            let _ = cg.insert_checkpoint(cp_a).expect("should insert cp");
            let _ = cg
                .insert_tx(tx_a.clone(), TxHeight::Confirmed(0))
                .expect("should insert tx");
            let _ = cg
                .insert_tx(tx_b.clone(), TxHeight::Unconfirmed)
                .expect("should insert tx");
            cg
        };
        let cg2 = {
            let mut cg = ChainGraph::default();
            let _ = cg
                .insert_tx(tx_b2.clone(), TxHeight::Unconfirmed)
                .expect("should insert tx");
            cg
        };

        let changeset = ChangeSet::<TxHeight> {
            chain: sparse_chain::ChangeSet {
                checkpoints: Default::default(),
                txids: [
                    (tx_b.txid(), None),
                    (tx_b2.txid(), Some(TxHeight::Unconfirmed)),
                ]
                .into(),
            },
            graph: Additions {
                tx: [tx_b2.clone()].into(),
                txout: [].into(),
            },
        };
        assert_eq!(
            cg1.determine_changeset(&cg2),
            Ok(changeset.clone()),
            "tx should be evicted from mempool"
        );

        cg1.apply_changeset(changeset).expect("should apply");
    }

    {
        let cg1 = {
            let mut cg = ChainGraph::default();
            let _ = cg.insert_checkpoint(cp_a).expect("should insert cp");
            let _ = cg.insert_checkpoint(cp_b).expect("should insert cp");
            let _ = cg
                .insert_tx(tx_a.clone(), TxHeight::Confirmed(0))
                .expect("should insert tx");
            let _ = cg
                .insert_tx(tx_b.clone(), TxHeight::Confirmed(1))
                .expect("should insert tx");
            cg
        };
        let cg2 = {
            let mut cg = ChainGraph::default();
            let _ = cg
                .insert_tx(tx_b2.clone(), TxHeight::Unconfirmed)
                .expect("should insert tx");
            cg
        };
        assert_eq!(
            cg1.determine_changeset(&cg2),
            Err(UpdateFailure::UnresolvableConflict(UnresolvableConflict {
                already_confirmed_tx: (TxHeight::Confirmed(1), tx_b.txid()),
                update_tx: (TxHeight::Unconfirmed, tx_b2.txid()),
            })),
            "fail if tx is evicted from valid block"
        );
    }

    {
        // Given 2 blocks `{A, B}`, and an update that invalidates block B with
        // `{A, B'}`, we expect txs that exist in `B` that conflicts with txs
        // introduced in the update to be successfully evicted.
        let mut cg1 = {
            let mut cg = ChainGraph::default();
            let _ = cg.insert_checkpoint(cp_a).expect("should insert cp");
            let _ = cg.insert_checkpoint(cp_b).expect("should insert cp");
            let _ = cg
                .insert_tx(tx_a.clone(), TxHeight::Confirmed(0))
                .expect("should insert tx");
            let _ = cg
                .insert_tx(tx_b.clone(), TxHeight::Confirmed(1))
                .expect("should insert tx");
            cg
        };
        let cg2 = {
            let mut cg = ChainGraph::default();
            let _ = cg.insert_checkpoint(cp_a).expect("should insert cp");
            let _ = cg.insert_checkpoint(cp_b2).expect("should insert cp");
            let _ = cg
                .insert_tx(tx_b2.clone(), TxHeight::Unconfirmed)
                .expect("should insert tx");
            cg
        };

        let changeset = ChangeSet::<TxHeight> {
            chain: sparse_chain::ChangeSet {
                checkpoints: [(1, Some(h!("B'")))].into(),
                txids: [
                    (tx_b.txid(), None),
                    (tx_b2.txid(), Some(TxHeight::Unconfirmed)),
                ]
                .into(),
            },
            graph: Additions {
                tx: [tx_b2.clone()].into(),
                txout: [].into(),
            },
        };
        assert_eq!(
            cg1.determine_changeset(&cg2),
            Ok(changeset.clone()),
            "tx should be evicted from B",
        );

        cg1.apply_changeset(changeset).expect("should apply");
    }
}

#[test]
fn update_missing_full_tx_errors() {
    let mut cg = ChainGraph::default();
    let chain_changeset = changeset! {
        checkpoints: [
            (0, Some(h!("A")))
        ],
        txids: [
            (h!("a1"), Some(TxHeight::Confirmed(0))),
            (h!("a2"), Some(TxHeight::Unconfirmed))
        ]
    };
    let changeset = ChangeSet {
        chain: chain_changeset,
        graph: tx_graph::Additions::default(),
    };
    let mut expected = HashSet::<Txid>::new();
    expected.insert(h!("a1"));
    expected.insert(h!("a2"));
    assert_eq!(
        cg.apply_changeset(changeset.clone()),
        Err((changeset, expected))
    );
}

#[test]
fn update_not_including_tx_already_in_graph_is_ok() {
    let mut cg = ChainGraph::default();
    let tx_a = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };
    let chain_changeset = changeset! {
        checkpoints: [ (0, Some(h!("A"))) ],
        txids: [
            (tx_a.txid(), Some(TxHeight::Confirmed(0)))
        ]
    };

    let mut graph_additions = tx_graph::Additions::default();
    graph_additions.tx.insert(tx_a.clone());
    let changeset = ChangeSet {
        chain: chain_changeset,
        graph: graph_additions,
    };

    assert_eq!(cg.apply_changeset(changeset), Ok(()));

    // reorg the tx to a different height and provide changeset without the full tx.
    let chain_changeset = changeset! {
        checkpoints: [(0, Some(h!("A'")))],
        txids: [
            (tx_a.txid(), Some(TxHeight::Unconfirmed))
        ]
    };

    let changeset = ChangeSet {
        chain: chain_changeset,
        graph: tx_graph::Additions::default(),
    };

    assert_eq!(cg.apply_changeset(changeset), Ok(()));
}

#[test]
fn chain_graph_inflate_changeset() {
    let mut cg = ChainGraph::default();
    let tx_a = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };
    let tx_b = Transaction {
        version: 0x02,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };

    let chain_changeset = changeset! {
        checkpoints: [ (0, Some(h!("A"))) ],
        txids: [
            (tx_a.txid(), Some(TxHeight::Confirmed(0))),
            (tx_b.txid(), Some(TxHeight::Confirmed(0)))
        ]
    };

    let mut expected_missing = HashSet::new();
    expected_missing.insert(tx_a.txid());
    expected_missing.insert(tx_b.txid());

    assert_eq!(
        cg.inflate_changeset(chain_changeset.clone(), vec![]),
        Err(InflateFailure::Missing(expected_missing.clone()))
    );

    expected_missing.remove(&tx_b.txid());
    assert_eq!(
        cg.inflate_changeset(chain_changeset.clone(), vec![tx_b.clone()]),
        Err(InflateFailure::Missing(expected_missing))
    );

    let mut additions = tx_graph::Additions::default();
    additions.tx.insert(tx_a.clone());
    additions.tx.insert(tx_b.clone());
    let changeset = cg.inflate_changeset(chain_changeset.clone(), vec![tx_a, tx_b]);
    assert_eq!(
        changeset,
        Ok(ChangeSet {
            chain: chain_changeset,
            graph: additions
        })
    );

    assert_eq!(cg.apply_changeset(changeset.unwrap()), Ok(()))
}

#[test]
fn test_get_tx_in_chain() {
    let mut cg = ChainGraph::default();
    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut::default()],
    };

    let _ = cg.insert_tx(tx.clone(), TxHeight::Unconfirmed).unwrap();
    assert_eq!(
        cg.get_tx_in_chain(tx.txid()),
        Some((&TxHeight::Unconfirmed, &tx))
    );
}

#[test]
fn test_iterate_transactions() {
    let mut cg = ChainGraph::default();
    let txs = (0..3)
        .map(|i| Transaction {
            version: i,
            lock_time: PackedLockTime(0),
            input: vec![],
            output: vec![TxOut::default()],
        })
        .collect::<Vec<_>>();
    let _ = cg
        .insert_checkpoint(BlockId {
            height: 1,
            hash: h!("A"),
        })
        .unwrap();
    let _ = cg
        .insert_tx(txs[0].clone(), TxHeight::Confirmed(1))
        .unwrap();
    let _ = cg.insert_tx(txs[1].clone(), TxHeight::Unconfirmed).unwrap();
    let _ = cg
        .insert_tx(txs[2].clone(), TxHeight::Confirmed(0))
        .unwrap();

    assert_eq!(
        cg.transactions_in_chain().collect::<Vec<_>>(),
        vec![
            (&TxHeight::Confirmed(0), &txs[2]),
            (&TxHeight::Confirmed(1), &txs[0]),
            (&TxHeight::Unconfirmed, &txs[1]),
        ]
    );
}
