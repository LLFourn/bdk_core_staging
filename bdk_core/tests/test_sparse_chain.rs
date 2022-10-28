use bdk_core::*;
use testing::*;

pub fn sdfsdsf() {}

#[test]
fn check_last_valid_rules() {
    let mut chain = SparseChain::default();

    fn gen_update(checkpoints: impl IntoIterator<Item = BlockId>) -> Update {
        Update {
            checkpoints: checkpoints
                .into_iter()
                .map(|cp| (cp.height, cp.hash))
                .collect(),
            txids: [].into(),
        }
    }

    assert_eq!(
        chain
            .apply_update(gen_update([gen_block_id(0, 0)]))
            .expect("add first tip should succeed"),
        ChangeSet {
            checkpoints: [(0, Change::new_insertion(gen_hash(0)))].into(),
            ..Default::default()
        },
    );

    chain
        .apply_update(gen_update([gen_block_id(0, 0), gen_block_id(1, 1)]))
        .expect("applying second tip on top of first should succeed");

    assert_eq!(
        chain
            .apply_update(gen_update([gen_block_id(2, 2)]))
            .expect_err("applying tip without specifying last valid should fail"),
        UpdateFailure::NotConnected,
    );

    assert_eq!(
        chain
            .apply_update(gen_update([gen_block_id(1, 2), gen_block_id(3, 3,)]))
            .expect_err("apply tip, while specifying non-existant last_valid should fail"),
        UpdateFailure::NotConnected
    );

    {
        let mut chain = chain.clone();
        assert_eq!(
            chain
                .apply_update(gen_update([gen_block_id(0, 3), gen_block_id(1, 4)]))
                .expect("update connects to empty chain"),
            ChangeSet {
                checkpoints: [
                    (0, Change::new(Some(gen_hash(0)), Some(gen_hash(3)))),
                    (1, Change::new(Some(gen_hash(1)), Some(gen_hash(4)))),
                ]
                .into(),
                ..Default::default()
            }
        )
    }
}

// #[test]
// fn check_invalidate_rules() {
//     let mut chain = SparseChain::default();

//     // add one checkpoint
//     assert_eq!(
//         chain
//             .apply_update(Update::new(None, gen_block_id(1, 1)))
//             .expect("should succeed"),
//         ChangeSet {
//             checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
//             ..Default::default()
//         },
//     );

//     // when we are invalidating the one and only checkpoint, `last_valid` should be `None`
//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 invalidate: Some(gen_block_id(1, 1)),
//                 ..Update::new(Some(gen_block_id(1, 1)), gen_block_id(1, 2))
//             })
//             .expect_err("update should fail when invalidate does not directly preceed last_valid"),
//         UpdateFailure::Stale {
//             got_last_valid: Some(gen_block_id(1, 1)),
//             expected_last_valid: None,
//         },
//     );
//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 invalidate: Some(gen_block_id(1, 1)),
//                 ..Update::new(None, gen_block_id(1, 2))
//             })
//             .expect("invalidate should succeed"),
//         ChangeSet {
//             checkpoints: [(1, Change::new(Some(gen_hash(1)), Some(gen_hash(2))))].into(),
//             ..Default::default()
//         }
//     );

//     // add two checkpoints
//     assert_eq!(
//         chain
//             .apply_update(Update::new(Some(gen_block_id(1, 2)), gen_block_id(2, 3)))
//             .expect("update should succeed"),
//         ChangeSet {
//             checkpoints: [(2, Change::new_insertion(gen_hash(3)))].into(),
//             ..Default::default()
//         }
//     );
//     assert_eq!(
//         chain
//             .apply_update(Update::new(Some(gen_block_id(2, 3)), gen_block_id(3, 4)))
//             .expect("update should succeed"),
//         ChangeSet {
//             checkpoints: [(3, Change::new_insertion(gen_hash(4)))].into(),
//             ..Default::default()
//         },
//     );

//     // `invalidate` should directly follow `last_valid`
//     assert_eq!(
//         chain.apply_update(Update {
//             invalidate: Some(gen_block_id(3, 4)),
//             ..Update::new(Some(gen_block_id(1, 2)), gen_block_id(3, 5))
//         }).expect_err("update should fail when checkpoint directly following last_valid is not invalidate"),
//         UpdateFailure::Stale {
//             got_last_valid: Some(gen_block_id(1, 2)),
//             expected_last_valid: Some(gen_block_id(2, 3)),
//         }
//     );
//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 invalidate: Some(gen_block_id(3, 4)),
//                 ..Update::new(Some(gen_block_id(2, 3)), gen_block_id(3, 5))
//             })
//             .expect("should succeed"),
//         ChangeSet {
//             checkpoints: [(3, Change::new(Some(gen_hash(4)), Some(gen_hash(5))))].into(),
//             ..Default::default()
//         }
//     );
// }

// #[test]
// fn apply_tips() {
//     let mut chain = SparseChain::default();

//     // gen 10 checkpoints
//     let mut last_valid = None;
//     for i in 0..10 {
//         let new_tip = gen_block_id(i, i as _);
//         chain
//             .apply_update(Update::new(last_valid, new_tip))
//             .expect("should succeed");
//         last_valid = Some(new_tip);
//     }

//     // repeated last tip should succeed
//     assert_eq!(
//         chain
//             .apply_update(Update::new(last_valid, last_valid.unwrap()))
//             .expect("repeated last_tip should succeed"),
//         ChangeSet::default(),
//     );

//     // ensure state of sparsechain is correct
//     chain
//         .range_checkpoints(..)
//         .zip(0..)
//         .for_each(|(block_id, exp_height)| {
//             assert_eq!(block_id, gen_block_id(exp_height, exp_height as _))
//         });
// }

// #[test]
// fn checkpoint_limit_is_respected() {
//     let mut chain = SparseChain::default();
//     chain.set_checkpoint_limit(Some(5));

//     // gen 10 checkpoints
//     let mut last_valid = None;
//     for i in 0..10 {
//         let new_tip = gen_block_id(i, i as _);

//         let changes = chain
//             .apply_update(Update {
//                 txids: [(gen_hash(i as _), TxHeight::Confirmed(i))].into(),
//                 ..Update::new(last_valid, new_tip)
//             })
//             .expect("should succeed");

//         assert_eq!(
//             changes,
//             ChangeSet {
//                 checkpoints: if i < 5 {
//                     [(i, Change::new_insertion(gen_hash(i as _)))].into()
//                 } else {
//                     [
//                         (i, Change::new_insertion(gen_hash(i as _))),
//                         (i - 5, Change::new_removal(gen_hash((i - 5) as _))),
//                     ]
//                     .into()
//                 },
//                 txids: [(
//                     gen_hash(i as _),
//                     Change::new_insertion(TxHeight::Confirmed(i))
//                 )]
//                 .into(),
//             }
//         );

//         last_valid = Some(new_tip);
//     }

//     assert_eq!(chain.iter_confirmed_txids().count(), 10);
//     assert_eq!(chain.range_checkpoints(..).count(), 5);
// }

// #[test]
// fn add_txids() {
//     let mut chain = SparseChain::default();

//     let txids = (0..100)
//         .map(gen_hash::<Txid>)
//         .map(|txid| (txid, TxHeight::Confirmed(1)))
//         .collect::<HashMap<_, _>>();

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: txids.clone(),
//                 ..Update::new(None, gen_block_id(1, 1))
//             })
//             .expect("add many txs in single checkpoint should succeed"),
//         ChangeSet {
//             checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
//             txids: txids
//                 .iter()
//                 .map(|(txid, height)| (*txid, Change::new_insertion(*height)))
//                 .collect(),
//         }
//     );

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(2), TxHeight::Confirmed(3))]
//                     .into_iter()
//                     .collect(),
//                 ..Update::new(Some(gen_block_id(1, 1)), gen_block_id(2, 2))
//             })
//             .expect_err("update that adds tx with height greater than hew tip should fail"),
//         UpdateFailure::Bogus(BogusReason::TxHeightGreaterThanTip {
//             new_tip: gen_block_id(2, 2),
//             tx: (gen_hash(2), TxHeight::Confirmed(3)),
//         })
//     );
// }

// #[test]
// fn add_txs_of_same_height_with_different_updates() {
//     let mut chain = SparseChain::default();
//     let block = gen_block_id(0, 0);

//     // add one block
//     assert_eq!(
//         chain
//             .apply_update(Update::new(None, block))
//             .expect("should succeed"),
//         ChangeSet {
//             checkpoints: [(0, Change::new_insertion(gen_hash(0)))].into(),
//             ..Default::default()
//         }
//     );

//     // add txs of same height with different updates
//     (0..100).for_each(|i| {
//         assert_eq!(
//             chain
//                 .apply_update(Update {
//                     txids: [(gen_hash(i as _), TxHeight::Confirmed(0))].into(),
//                     ..Update::new(Some(block), block)
//                 })
//                 .expect("should succeed"),
//             ChangeSet {
//                 txids: [(
//                     gen_hash(i as _),
//                     Change::new_insertion(TxHeight::Confirmed(0))
//                 )]
//                 .into(),
//                 ..Default::default()
//             }
//         );
//     });

//     assert_eq!(chain.iter_txids().count(), 100);
//     assert_eq!(chain.iter_confirmed_txids().count(), 100);
//     assert_eq!(chain.iter_mempool_txids().count(), 0);
//     assert_eq!(chain.range_checkpoints(..).count(), 1);
// }

// #[test]
// fn confirm_tx() {
//     let mut chain = SparseChain::default();

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [
//                     (gen_hash(10), TxHeight::Unconfirmed),
//                     (gen_hash(20), TxHeight::Unconfirmed),
//                 ]
//                 .into(),
//                 ..Update::new(None, gen_block_id(1, 1))
//             })
//             .expect("adding two txs from mempool should succeed"),
//         ChangeSet {
//             checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
//             txids: [
//                 (gen_hash(10), Change::new_insertion(TxHeight::Unconfirmed)),
//                 (gen_hash(20), Change::new_insertion(TxHeight::Unconfirmed))
//             ]
//             .into()
//         }
//     );

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(10), TxHeight::Confirmed(0))].into(),
//                 ..Update::new(Some(gen_block_id(1, 1)), gen_block_id(1, 1))
//             })
//             .expect("it should be okay to confirm tx into block before last_valid (partial sync)"),
//         ChangeSet {
//             txids: [(
//                 gen_hash(10),
//                 Change::new(Some(TxHeight::Unconfirmed), Some(TxHeight::Confirmed(0)))
//             )]
//             .into(),
//             ..Default::default()
//         }
//     );
//     assert_eq!(chain.iter_txids().count(), 2);
//     assert_eq!(chain.iter_confirmed_txids().count(), 1);
//     assert_eq!(chain.iter_mempool_txids().count(), 1);

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(20), TxHeight::Confirmed(2))].into(),
//                 ..Update::new(Some(gen_block_id(1, 1)), gen_block_id(2, 2))
//             })
//             .expect("it should be okay to confirm tx into the tip introduced"),
//         ChangeSet {
//             checkpoints: [(2, Change::new_insertion(gen_hash(2)))].into(),
//             txids: [(
//                 gen_hash(20),
//                 Change::new(Some(TxHeight::Unconfirmed), Some(TxHeight::Confirmed(2)))
//             )]
//             .into(),
//         }
//     );
//     assert_eq!(chain.iter_txids().count(), 2);
//     assert_eq!(chain.iter_confirmed_txids().count(), 2);
//     assert_eq!(chain.iter_mempool_txids().count(), 0);

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(10), TxHeight::Unconfirmed)].into(),
//                 ..Update::new(Some(gen_block_id(2, 2)), gen_block_id(2, 2))
//             })
//             .expect_err("tx cannot be unconfirmed without invalidate"),
//         UpdateFailure::Inconsistent {
//             inconsistent_txid: gen_hash(10),
//             original_height: TxHeight::Confirmed(0),
//             update_height: TxHeight::Unconfirmed,
//         }
//     );

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(20), TxHeight::Confirmed(3))].into(),
//                 ..Update::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
//             })
//             .expect_err("tx cannot move forward in blocks without invalidate"),
//         UpdateFailure::Inconsistent {
//             inconsistent_txid: gen_hash(20),
//             original_height: TxHeight::Confirmed(2),
//             update_height: TxHeight::Confirmed(3),
//         },
//     );

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(20), TxHeight::Confirmed(1))].into(),
//                 ..Update::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
//             })
//             .expect_err("tx cannot move backwards in blocks without invalidate"),
//         UpdateFailure::Inconsistent {
//             inconsistent_txid: gen_hash(20),
//             original_height: TxHeight::Confirmed(2),
//             update_height: TxHeight::Confirmed(1),
//         },
//     );

//     assert_eq!(
//         chain
//             .apply_update(Update {
//                 txids: [(gen_hash(20), TxHeight::Confirmed(2))].into(),
//                 ..Update::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
//             })
//             .expect("update can introduce already-existing tx"),
//         ChangeSet {
//             checkpoints: [(3, Change::new_insertion(gen_hash(3)))].into(),
//             ..Default::default()
//         }
//     );
//     assert_eq!(chain.iter_txids().count(), 2);
//     assert_eq!(chain.iter_confirmed_txids().count(), 2);
//     assert_eq!(chain.iter_mempool_txids().count(), 0);
// }
