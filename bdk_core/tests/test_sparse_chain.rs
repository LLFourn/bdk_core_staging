use bdk_core::*;
mod checkpoint_gen;
use bitcoin::{hashes::Hash, BlockHash, OutPoint};
use checkpoint_gen::{CheckpointGen, ISpec, OSpec, TxSpec};

#[test]
fn invalid_tx_confirmation_time() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let update = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 1)],
            confirmed_at: Some(2),
        }],
        1,
    );

    assert_eq!(
        chain.apply_checkpoint(update.clone()),
        ApplyResult::Stale(StaleReason::TxidHeightGreaterThanNewTip {
            tip: update.new_tip,
            txid: update.txids[0]
        }),
    );

    assert_eq!(chain.iter_checkpoints(..).count(), 0);
    assert_eq!(chain.iter_txids().count(), 0);
}

#[test]
fn out_of_order_tx_is_before_first_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
            &mut graph,
            vec![TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(2_000, 1)],
                confirmed_at: Some(1),
            },],
            1,
        )),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
            &mut graph,
            vec![TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(2_000, 1)],
                confirmed_at: Some(0),
            },],
            2,
        )),
        ApplyResult::Ok
    );
}

#[test]
fn checkpoint_limit_is_applied() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    // chain.set_checkpoint_limit(5);

    for i in 0..10 {
        assert_eq!(
            chain.apply_checkpoint(checkpoint_gen.create_update(
                &mut graph,
                vec![TxSpec {
                    inputs: vec![ISpec::Other],
                    outputs: vec![OSpec::Mine(2_000, i)],
                    confirmed_at: Some(i as u32),
                },],
                i as u32,
            )),
            ApplyResult::Ok
        );
    }

    assert_eq!(chain.iter_confirmed_txids().count(), 10);
    assert_eq!(chain.iter_checkpoints(..).count(), 10);
}

#[test]
fn many_transactions_in_the_same_height() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let txs = (0..100)
        .map(|_| TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(1),
        })
        .collect();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(&mut graph, txs, 1,)),
        ApplyResult::Ok
    );
}

#[test]
fn same_checkpoint_twice_should_be_stale() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let update = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.apply_checkpoint(update), ApplyResult::Ok);
}

#[test]
fn adding_checkpoint_where_new_tip_is_base_tip_is_fine() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let mut update = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.last_valid = Some(update.new_tip);
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 1);
}

#[test]
fn adding_checkpoint_which_contains_nothing_new_should_create_single_empty_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let mut update = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.last_valid = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 1,
        ..Default::default()
    };
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 2);

    update.last_valid = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 2,
        ..Default::default()
    };
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 3);
    assert_eq!(chain.iter_checkpoints(..).next().unwrap().height, 0);
    assert_eq!(chain.iter_checkpoints(..).last().unwrap().height, 2);
}

#[test]
fn adding_checkpoint_where_tx_conftime_has_changed() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let mut update = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.last_valid = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 1,
        ..Default::default()
    };
    update.txids[0].1 = Some(1);
    assert!(matches!(
        chain.apply_checkpoint(update.clone()),
        ApplyResult::Inconsistent { .. }
    ));
    assert_eq!(chain.iter_checkpoints(..).count(), 1);
}

#[test]
fn invalidte_first_and_only_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let update1 = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(1),
        }],
        1,
    );
    assert_eq!(chain.apply_checkpoint(update1.clone()), ApplyResult::Ok);

    let update2 = CheckpointCandidate {
        last_valid: Some(BlockId {
            height: 1,
            hash: BlockHash::from_slice(&[1_u8; 32]).unwrap(),
        }),
        ..checkpoint_gen.create_update(
            &mut graph,
            vec![TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(2_900, 0)],
                confirmed_at: Some(2),
            }],
            2,
        )
    };
    assert_eq!(
        chain.apply_checkpoint(update2.clone()),
        ApplyResult::Stale(StaleReason::LastValidHashNotMatching {
            got: chain.latest_checkpoint().map(|b| b.hash),
            expected: update2.last_valid.unwrap(),
        })
    );

    let update3 = CheckpointCandidate {
        last_valid: None,
        invalidate: Some(update1.new_tip),
        ..update2.clone()
    };
    assert_eq!(chain.apply_checkpoint(update3.clone()), ApplyResult::Ok);
    println!(
        "confirmed: {:#?}",
        chain.iter_confirmed_txids().collect::<Vec<_>>()
    );
    assert_eq!(chain.iter_confirmed_txids().count(), 1);

    let tx = chain
        .iter_confirmed_txids()
        .next()
        .map(|(_, txid)| graph.tx(txid))
        .flatten()
        .unwrap();
    assert_eq!(tx.output(0).unwrap().value, 2_900);
}

#[test]
fn checkpoints_at_same_height_with_different_tx_applied_one_after_the_other() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let update1 = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update1.clone()), ApplyResult::Ok);

    let mut update2 = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    update2.last_valid = Some(update1.new_tip);
    assert_eq!(chain.apply_checkpoint(update2.clone()), ApplyResult::Ok);

    assert_eq!(chain.iter_checkpoints(..).count(), 1);
    assert_eq!(chain.iter_confirmed_txids().count(), 2);
}

#[test]
fn output_is_spent() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();

    let first = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    let txid = first.txids[0].0;

    let second = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint { txid, vout: 0 }), ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(1),
        }],
        1,
    );

    assert_eq!(chain.apply_checkpoint(first.clone()), ApplyResult::Ok);
    assert_eq!(chain.apply_checkpoint(second.clone()), ApplyResult::Ok);
    let outspend_set = graph
        .outspend(&OutPoint { txid, vout: 0 })
        .expect("should have outspend");
    assert!(outspend_set.contains(&second.txids[0].0));
}

// TODO: Implement consistency detection
// #[test]
// fn spent_outpoint_doesnt_exist_but_tx_does() {
//     let mut checkpoint_gen = CheckpointGen::new();
//     let mut chain = SparseChain::default();
//     let mut graph = TxGraph::default();

//     let first = checkpoint_gen.create_update(
//         &mut graph,
//         vec![TxSpec {
//             inputs: vec![ISpec::Other],
//             outputs: vec![OSpec::Mine(1_000, 0)],
//             confirmed_at: Some(0),
//         }],
//         0,
//     );

//     assert_eq!(chain.apply_checkpoint(first.clone()), ApplyResult::Ok);

//     let spends_impossible_output = checkpoint_gen.create_update(
//         &mut graph,
//         vec![TxSpec {
//             inputs: vec![ISpec::Explicit(OutPoint {
//                 txid: first.txids[0].0,
//                 vout: 1,
//             })],
//             outputs: vec![OSpec::Mine(1_000, 0)],
//             confirmed_at: Some(0),
//         }],
//         0,
//     );

//     assert!(matches!(
//         chain.apply_checkpoint(spends_impossible_output),
//         ApplyResult::Inconsistent { .. }
//     ));
// }

// TODO: add test for adding the target

#[test]
fn empty_checkpoint_doesnt_get_removed() {
    let mut chain = SparseChain::default();
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![],
            last_valid: None,
            invalidate: None,
            new_tip: BlockId {
                height: 0,
                ..Default::default()
            },
        }),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.latest_checkpoint(),
        Some(BlockId {
            height: 0,
            ..Default::default()
        })
    );
}
