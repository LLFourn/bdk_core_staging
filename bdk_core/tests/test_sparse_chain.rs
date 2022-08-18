use bdk_core::*;
mod checkpoint_gen;
use bitcoin::{BlockHash, OutPoint};
use checkpoint_gen::{CheckpointGen, ISpec, OSpec, TxSpec};

#[test]
fn two_checkpoints_then_merge() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
            vec![
                TxSpec {
                    inputs: vec![ISpec::Other],
                    outputs: vec![OSpec::Mine(2_000, 0)],
                    confirmed_at: Some(1),
                },
                TxSpec {
                    inputs: vec![ISpec::Other],
                    outputs: vec![OSpec::Mine(1_000, 1)],
                    confirmed_at: Some(0),
                },
            ],
            1,
        )),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
            vec![
                TxSpec {
                    inputs: vec![ISpec::Other],
                    outputs: vec![OSpec::Mine(3_000, 2)],
                    confirmed_at: Some(2),
                },
                TxSpec {
                    inputs: vec![ISpec::Other],
                    outputs: vec![OSpec::Mine(4_000, 3)],
                    confirmed_at: Some(3),
                },
            ],
            3,
        )),
        ApplyResult::Ok
    );

    // there is no checkpoint here
    chain.merge_checkpoint(0);
    assert_eq!(chain.iter_checkpoints(..).count(), 2);

    chain.merge_checkpoint(1);
    assert_eq!(
        chain.iter_checkpoints(..).count(),
        1,
        "only one checkpoint after merge"
    );
    assert_eq!(
        chain
            .checkpoint_txids(chain.checkpoint_at(3).unwrap())
            .count(),
        4
    );

    chain.merge_checkpoint(1);
    assert_eq!(
        chain.iter_checkpoints(..).count(),
        1,
        "merging last checpoint has no affect"
    );
    assert_eq!(
        chain
            .checkpoint_txids(chain.checkpoint_at(3).unwrap())
            .count(),
        4
    );
}

#[test]
fn invalid_tx_confirmation_time() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
            vec![TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(2_000, 1)],
                confirmed_at: Some(2),
            },],
            1,
        )),
        ApplyResult::Ok
    );

    assert_eq!(chain.iter_checkpoints(..).count(), 1);
    assert_eq!(chain.iter_tx().count(), 0);
}

#[test]
fn out_of_order_tx_is_before_first_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(
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
    chain.set_checkpoint_limit(5);

    for i in 0..10 {
        assert_eq!(
            chain.apply_checkpoint(checkpoint_gen.create_update(
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

    assert_eq!(chain.iter_tx().count(), 10);
    assert_eq!(chain.iter_checkpoints(..).count(), 5);
}

#[test]
fn many_transactions_in_the_same_height() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let txs = (0..100)
        .map(|_| TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(1),
        })
        .collect();

    assert_eq!(
        chain.apply_checkpoint(checkpoint_gen.create_update(txs, 1,)),
        ApplyResult::Ok
    );
}

#[test]
fn same_checkpoint_twice_should_be_stale() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let update = checkpoint_gen.create_update(
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

    let mut update = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.base_tip = Some(update.new_tip);
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 1);
}

#[test]
fn adding_checkpoint_which_contains_nothing_new_should_create_single_empty_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let mut update = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.base_tip = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 1,
        ..Default::default()
    };
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 2);

    update.base_tip = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 2,
        ..Default::default()
    };
    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_checkpoints(..).count(), 2);
    assert_eq!(chain.iter_checkpoints(..).next().unwrap().height, 0);
    assert_eq!(chain.iter_checkpoints(..).last().unwrap().height, 2);
}

#[test]
fn adding_checkpoint_where_tx_conftime_has_changed() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let mut update = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update.clone()), ApplyResult::Ok);
    update.base_tip = Some(update.new_tip);
    update.new_tip = BlockId {
        height: 1,
        ..Default::default()
    };
    update.transactions[0].confirmation_time = Some(BlockTime { height: 1, time: 1 });
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

    let update1 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update1.clone()), ApplyResult::Ok);
    let mut update2 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(2_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    update2.base_tip = None;

    assert_eq!(
        chain.apply_checkpoint(update2.clone()),
        ApplyResult::Stale(StaleReason::BaseTipNotMatching {
            got: None,
            expected: chain.latest_checkpoint().unwrap()
        })
    );

    update2.invalidate = Some(update1.new_tip);

    assert_eq!(chain.apply_checkpoint(update2.clone()), ApplyResult::Ok);
    assert_eq!(chain.iter_tx().count(), 1);
    assert_eq!(chain.iter_tx().next().unwrap().1.tx.output[0].value, 2_900);
}

#[test]
fn simple_double_spend_inconsistency_check() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let first = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    let txid = first.transactions[0].tx.txid();

    let second = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint { txid, vout: 0 }), ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(1),
        }],
        1,
    );

    let third = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint { txid, vout: 0 }), ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 1)],
            confirmed_at: Some(1),
        }],
        1,
    );

    assert_eq!(chain.apply_checkpoint(first), ApplyResult::Ok);
    assert_eq!(chain.apply_checkpoint(second.clone()), ApplyResult::Ok);
    assert_eq!(
        chain.apply_checkpoint(third.clone()),
        ApplyResult::Inconsistent {
            txid: third.transactions[0].tx.txid(),
            conflicts_with: second.transactions[0].tx.txid(),
        }
    );
}

#[test]
fn checkpoints_at_same_height_with_different_tx_applied_one_after_the_other() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let update1 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(update1.clone()), ApplyResult::Ok);

    let mut update2 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_900, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    update2.base_tip = Some(update1.new_tip);
    assert_eq!(chain.apply_checkpoint(update2.clone()), ApplyResult::Ok);

    assert_eq!(chain.iter_checkpoints(..).count(), 1);
    assert_eq!(chain.iter_tx().count(), 2);
}

#[test]
fn output_is_spent() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let first = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    let txid = first.transactions[0].tx.txid();

    let second = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint { txid, vout: 0 }), ISpec::Other],
            outputs: vec![OSpec::Mine(2_000, 0)],
            confirmed_at: Some(1),
        }],
        1,
    );

    assert_eq!(chain.apply_checkpoint(first.clone()), ApplyResult::Ok);
    assert_eq!(chain.apply_checkpoint(second.clone()), ApplyResult::Ok);
    assert_eq!(
        chain.outspend(OutPoint {
            txid: txid,
            vout: 0
        }),
        Some(second.transactions[0].tx.txid())
    );
}

#[test]
fn spent_outpoint_doesnt_exist_but_tx_does() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();

    let first = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(first.clone()), ApplyResult::Ok);

    let spends_impossible_output = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint {
                txid: first.transactions[0].tx.txid(),
                vout: 1,
            })],
            outputs: vec![OSpec::Mine(1_000, 0)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert!(matches!(
        chain.apply_checkpoint(spends_impossible_output),
        ApplyResult::Inconsistent { .. }
    ));
}

// TODO: add test for adding the target

#[test]
fn empty_checkpoint_doesnt_get_removed() {
    let mut chain = SparseChain::default();
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            transactions: vec![],
            base_tip: None,
            invalidate: None,
            new_tip: BlockId {
                height: 0,
                hash: BlockHash::default(),
            },
        }),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.latest_checkpoint(),
        Some(BlockId {
            height: 0,
            hash: BlockHash::default()
        })
    );
}

#[test]
fn two_empty_checkpoints_get_merged() {
    let mut chain = SparseChain::default();
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            transactions: vec![],
            base_tip: None,
            invalidate: None,
            new_tip: BlockId::default(),
        }),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.latest_checkpoint(),
        Some(BlockId {
            height: 0,
            hash: BlockHash::default()
        })
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            transactions: vec![],
            base_tip: Some(BlockId::default()),
            invalidate: None,
            new_tip: BlockId {
                height: 1,
                ..Default::default()
            },
        }),
        ApplyResult::Ok
    );

    assert_eq!(
        chain.latest_checkpoint(),
        Some(BlockId {
            height: 1,
            hash: BlockHash::default()
        })
    );
    assert_eq!(chain.iter_checkpoints(..).count(), 1);
}
