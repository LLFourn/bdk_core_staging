#[macro_use]
mod common;
use bdk_core::tx_graph::{Additions, TxGraph};
use bitcoin::{OutPoint, Script, TxOut};

#[test]
fn simple_update() {
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
            assert!(graph.insert_txout(*outpoint, txout));
        }
        graph
    };

    let update = {
        let mut graph = TxGraph::default();
        for (outpoint, txout) in &update_ops {
            assert!(graph.insert_txout(*outpoint, &txout));
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

    graph.apply_additions(&additions);
    assert_eq!(graph.iter_all_txouts().count(), 3);
    assert_eq!(graph.iter_full_transactions().count(), 0);
    assert_eq!(graph.iter_partial_transactions().count(), 2);
}
