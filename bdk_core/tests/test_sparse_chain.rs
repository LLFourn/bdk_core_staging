use bdk_core::*;
use bitcoin::Txid;

fn chain_txs(chain: &SparseChain) -> (Vec<(TxHeight, Txid)>, Vec<(TxHeight, Txid)>) {
    chain
        .iter_txids()
        .partition::<Vec<_>, _>(|(h, _)| h.is_confirmed())
}

macro_rules! chain {
    ($([$($tt:tt)*]),*) => { chain!( checkpoints: [$([$($tt)*]),*] ) };
    (checkpoints: [ $([$height:expr, $block_hash:expr]),* ] $(,txids: [$(($txid_index:literal, $tx_height:expr)),*])?) => {{
        #[allow(unused_mut)]
        let mut chain = SparseChain::from_checkpoints(vec![$(($height, $block_hash).into()),*]);

        $(
            $(
                chain.insert_tx({
                    use bitcoin::hashes::Hash;
                    let txid_index: u32 = $txid_index;
                    Hash::hash(&txid_index.to_le_bytes()[..])
                }, $tx_height).unwrap();
            )*
        )?

        chain
    }};
}

macro_rules! h {
    ($index:literal) => {{
        use bitcoin::hashes::Hash;
        let hash_seed: u32 = $index;
        Hash::hash(&hash_seed.to_le_bytes()[..])
    }};
}

macro_rules! changeset {
    (
        checkpoints: [ $(( $height:expr, $cp_from:expr => $cp_to:expr )),* ],
        txids: [ $(( $txid:expr, $tx_from:expr => $tx_to:expr )),* ]

    ) => {{
        use bdk_core::collections::HashMap;
        #[allow(unused_mut)]
        ChangeSet {
            checkpoints: {
                let mut changes = HashMap::default();
                $(changes.insert($height, Change { from: $cp_from, to: $cp_to });)*
                changes
            },
            txids: {
                let mut changes = HashMap::default();
                $(changes.insert($txid, Change { from: $tx_from, to: $tx_to });)*
                changes
            }
        }
    }};
}

#[test]
fn add_first_checkpoint() {
    let chain = SparseChain::default();
    assert_eq!(
        chain.determine_changeset(&chain!([0, h!(0)])),
        Ok(changeset! {
            checkpoints: [(0, None => Some(h!(0)))],
            txids: []
        }),
        "add first tip"
    );
}

#[test]
fn add_second_tip() {
    let chain = chain!([0, h!(0)]);
    assert_eq!(
        chain.determine_changeset(&chain!([0, h!(0)], [1, h!(1)])),
        Ok(changeset! {
            checkpoints: [(1, None => Some(h!(1)))],
            txids: []
        }),
        "extend tip by one"
    );
}

#[test]
fn two_disjoint_chains_cannot_merge() {
    let chain1 = chain!([0, h!(0)]);
    let chain2 = chain!([1, h!(1)]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Err(UpdateFailure::NotConnected)
    );
}

#[test]
fn duplicate_chains_should_merge() {
    let chain1 = chain!([0, h!(0)]);
    let chain2 = chain!([0, h!(0)]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(ChangeSet::default())
    );
}

#[test]
fn duplicate_chains_with_txs_should_merge() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    let chain2 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(ChangeSet::default())
    );
}

#[test]
fn duplicate_chains_with_different_txs_should_merge() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    let chain2 = chain!(checkpoints: [[0,h!(0)]], txids: [(1, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [],
            txids: [(h!(1), None => Some(TxHeight::Confirmed(0)))]
        })
    );
}

#[test]
fn invalidate_first_and_only_checkpoint_without_tx_changes() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    let chain2 = chain!(checkpoints: [[0,h!(1)]], txids: [(0, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(0, Some(h!(0)) => Some(h!(1)))],
            txids: []
        })
    );
}

#[test]
fn invalidate_first_and_only_checkpoint_with_tx_move_forward() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    let chain2 = chain!(checkpoints: [[0,h!(1)],[1, h!(2)]], txids: [(0, TxHeight::Confirmed(1))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(0, Some(h!(0)) => Some(h!(1))), (1, None => Some(h!(2)))],
            txids: [(h!(0), Some(TxHeight::Confirmed(0)) => Some(TxHeight::Confirmed(1)))]
        })
    );
}

#[test]
fn invalidate_first_and_only_checkpoint_with_tx_move_backward() {
    let chain1 = chain!(checkpoints: [[1,h!(1)]], txids: [(0, TxHeight::Confirmed(1))]);
    let chain2 = chain!(checkpoints: [[0,h!(0)],[1, h!(2)]], txids: [(0, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(0, None => Some(h!(0))), (1, Some(h!(1)) => Some(h!(2)))],
            txids: [(h!(0), Some(TxHeight::Confirmed(1)) => Some(TxHeight::Confirmed(0)))]
        })
    );
}

#[test]
fn invalidate_first_and_only_checkpoint_with_tx_move_to_mempool() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    let chain2 = chain!(checkpoints: [[0,h!(1)]], txids: [(0, TxHeight::Unconfirmed)]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(0, Some(h!(0)) => Some(h!(1)))],
            txids: [(h!(0), Some(TxHeight::Confirmed(0)) => Some(TxHeight::Unconfirmed))]
        })
    );
}

#[test]
fn confirm_tx_without_extending_chain() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Unconfirmed)]);
    let chain2 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [],
            txids: [(h!(0), Some(TxHeight::Unconfirmed) => Some(TxHeight::Confirmed(0)))]
        })
    );
}

#[test]
fn confirm_tx_backwards_while_extending_chain() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Unconfirmed)]);
    let chain2 = chain!(checkpoints: [[0,h!(0)],[1,h!(1)]], txids: [(0, TxHeight::Confirmed(0))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(1, None => Some(h!(1)))],
            txids: [(h!(0), Some(TxHeight::Unconfirmed) => Some(TxHeight::Confirmed(0)))]
        })
    );
}

#[test]
fn confirm_tx_in_new_block() {
    let chain1 = chain!(checkpoints: [[0,h!(0)]], txids: [(0, TxHeight::Unconfirmed)]);
    let chain2 = chain!(checkpoints: [[0,h!(0)],[1,h!(1)]], txids: [(0, TxHeight::Confirmed(1))]);
    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(1, None => Some(h!(1)))],
            txids: [(h!(0), Some(TxHeight::Unconfirmed) => Some(TxHeight::Confirmed(1)))]
        })
    );
}

#[test]
fn merging_mempool_of_empty_chains_doesnt_fail() {
    let chain1 = chain!(checkpoints: [], txids: [(0, TxHeight::Unconfirmed)]);
    let chain2 = chain!(checkpoints: [], txids: [(1, TxHeight::Unconfirmed)]);

    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [],
            txids: [(h!(1), None => Some(TxHeight::Unconfirmed))]
        })
    );
}

#[test]
fn cannot_insert_confirmed_tx_without_checkpoints() {
    let mut chain = SparseChain::default();
    assert_eq!(
        chain.insert_tx(h!(0), TxHeight::Confirmed(0)),
        Err(InsertTxErr::TxTooHigh)
    );
}

#[test]
fn empty_chain_can_add_unconfirmed_transactions() {
    let chain1 = chain!(checkpoints: [[0, h!(0)]], txids: []);
    let chain2 = chain!(checkpoints: [], txids: [(0, TxHeight::Unconfirmed)]);

    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [],
            txids: [ (h!(0), None => Some(TxHeight::Unconfirmed)) ]
        })
    );
}

#[test]
fn can_update_with_shorter_chain() {
    let chain1 = chain!(checkpoints: [[1, h!(1)],[2, h!(2)]], txids: []);
    let chain2 = chain!(checkpoints: [[1, h!(1)]], txids: [(0, TxHeight::Confirmed(1))]);

    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [],
            txids: [(h!(0), None => Some(TxHeight::Confirmed(1)))]
        })
    )
}

#[test]
fn can_introduce_older_checkpoints() {
    let chain1 = chain!(checkpoints: [[2, h!(2)], [3, h!(3)]], txids: []);
    let chain2 = chain!(checkpoints: [[1, h!(1)], [2, h!(2)]], txids: []);

    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(1, None => Some(h!(1)))],
            txids: []
        })
    );
}

#[test]
fn fix_blockhash_before_agreement_point() {
    let chain1 = chain!(checkpoints: [[0, h!(0)], [1, h!(1)]], txids: []);
    let chain2 = chain!(checkpoints: [[0, h!(9)], [1, h!(1)]], txids: []);

    assert_eq!(
        chain1.determine_changeset(&chain2),
        Ok(changeset! {
            checkpoints: [(0, Some(h!(0)) => Some(h!(1)))],
            txids: []
        })
    )
}

// /// Update and chain does not connect:
// /// ```
// ///        | 0 | 1 | 2 | 3 | 4
// /// chain  |     B   C
// /// update | A   B       D
// /// ```
// /// This should fail as we cannot figure out whether C & D are on the same chain
// #[test]
// fn update_and_chain_does_not_connect() {
//     let mut chain = SparseChain::default();

//     assert_eq!(
//         chain.apply_update(&gen_update([gen_block_id(1, 1), gen_block_id(2, 2)])),
//         Ok(ChangeSet {
//             checkpoints: [
//                 (1, Change::new_insertion(gen_hash(1))),
//                 (2, Change::new_insertion(gen_hash(2)))
//             ]
//             .into(),
//             ..Default::default()
//         }),
//     );

//     assert_eq!(
//         chain.apply_update(&gen_update([
//             gen_block_id(0, 0),
//             gen_block_id(1, 1),
//             gen_block_id(3, 3)
//         ])),
//         Err(UpdateFailure::NotConnected),
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
//             .apply_update(&gen_update(
//                 last_valid.iter().chain(core::iter::once(&new_tip)).cloned(),
//             ))
//             .expect("should succeed");
//         last_valid = Some(new_tip);
//     }

//     // repeated last tip should succeed
//     assert_eq!(
//         chain
//             .apply_update(&gen_update(last_valid))
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
//             .apply_update(&Update {
//                 txs: [(gen_txid(i as _).into(), TxHeight::Confirmed(i))].into(),
//                 ..gen_update(last_valid.iter().chain(core::iter::once(&new_tip)).cloned())
//             })
//             .expect("should succeed");

//         assert_eq!(
//             changes,
//             ChangeSet {
//                 // TODO: Figure out whether checkpoint pruning should be represented in changeset
//                 checkpoints: //if i < 5 {
//                     [(i, Change::new_insertion(gen_hash(i as _)))].into(),
//                 // } else {
//                 //     [
//                 //         (i, Change::new_insertion(gen_hash(i as _))),
//                 //         (i - 5, Change::new_removal(gen_hash((i - 5) as _))),
//                 //     ]
//                 //     .into()
//                 // },
//                 txids: [(
//                     gen_hash(i as _),
//                     Change::new_insertion(TxHeight::Confirmed(i))
//                 )]
//                 .into(),
//             }
//         );

//         last_valid = Some(new_tip);
//     }

//     assert_eq!(chain.iter_txids().count(), 10);
//     assert_eq!(chain.range_checkpoints(..).count(), 5);
// }

// #[test]
// fn add_txids() {
//     let mut chain = SparseChain::default();

//     let update = Update {
//         txs: (0..100)
//             .map(gen_hash::<Txid>)
//             .map(|txid| (txid.into(), TxHeight::Confirmed(1)))
//             .collect(),
//         ..gen_update([gen_block_id(1, 1)])
//     };

//     assert_eq!(
//         chain.apply_update(&update),
//         Ok(ChangeSet {
//             checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
//             txids: update
//                 .txs
//                 .iter()
//                 .map(|(txid, height)| (txid.txid(), Change::new_insertion(*height)))
//                 .collect(),
//         }),
//         "add many txs in single checkpoint should succeed",
//     );
// }

// #[test]
// fn add_txs_of_same_height_with_different_updates() {
//     let mut chain = SparseChain::default();
//     let block = gen_block_id(0, 0);

//     // add one block
//     assert_eq!(
//         chain.apply_update(&gen_update([block])),
//         Ok(ChangeSet {
//             checkpoints: [(0, Change::new_insertion(gen_hash(0)))].into(),
//             ..Default::default()
//         }),
//         "should succeed",
//     );

//     // add txs of same height with different updates
//     (0..100).for_each(|i| {
//         assert_eq!(
//             chain
//                 .apply_update(&Update {
//                     txs: [(gen_txid(i as _).into(), TxHeight::Confirmed(0))].into(),
//                     ..gen_update([block])
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
//     assert_eq!(chain.range_checkpoints(..).count(), 1);
// }

// #[test]
// fn confirm_tx() {
//     let mut chain = SparseChain::default();

//     assert_eq!(
//         chain
//             .apply_update(&Update {
//                 txs: [
//                     (gen_txid(10).into(), TxHeight::Unconfirmed),
//                     (gen_txid(20).into(), TxHeight::Unconfirmed),
//                 ]
//                 .into(),
//                 ..gen_update([gen_block_id(1, 1)])
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
//             .apply_update(&Update {
//                 txs: [(gen_txid(10).into(), TxHeight::Confirmed(0))].into(),
//                 ..gen_update([gen_block_id(1, 1), gen_block_id(1, 1)])
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
//     let (confirmed, unconfirmed) = chain_txs(&chain);
//     assert_eq!(confirmed.len(), 1);
//     assert_eq!(unconfirmed.len(), 1);

//     assert_eq!(
//         chain
//             .apply_update(&Update {
//                 txs: [(gen_txid(20).into(), TxHeight::Confirmed(2))].into(),
//                 ..gen_update([gen_block_id(1, 1), gen_block_id(2, 2)])
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
//     let (confirmed, unconfirmed) = chain_txs(&chain);
//     assert_eq!(confirmed.len(), 2);
//     assert_eq!(unconfirmed.len(), 0);

//     assert_eq!(
//         chain
//             .apply_update(&Update {
//                 txs: [(gen_txid(10).into(), TxHeight::Unconfirmed)].into(),
//                 ..gen_update([gen_block_id(2, 2), gen_block_id(2, 2)])
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
//             .apply_update(&Update {
//                 txs: [(gen_txid(20).into(), TxHeight::Confirmed(3))].into(),
//                 ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
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
//             .apply_update(&Update {
//                 txs: [(gen_txid(20).into(), TxHeight::Confirmed(1))].into(),
//                 ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
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
//             .apply_update(&Update {
//                 txs: [(gen_txid(20).into(), TxHeight::Confirmed(2))].into(),
//                 ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
//             })
//             .expect("update can introduce already-existing tx"),
//         ChangeSet {
//             checkpoints: [(3, Change::new_insertion(gen_hash(3)))].into(),
//             ..Default::default()
//         }
//     );
//     let (confirmed, unconfirmed) = chain_txs(&chain);
//     assert_eq!(confirmed.len(), 2);
//     assert_eq!(unconfirmed.len(), 0);
// }
