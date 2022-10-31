use bdk_core::*;
use bitcoin::Txid;
use testing::*;

fn gen_update(checkpoints: impl IntoIterator<Item = BlockId>) -> Update {
    Update {
        checkpoints: checkpoints
            .into_iter()
            .map(|cp| (cp.height, cp.hash))
            .collect(),
        txs: [].into(),
    }
}

fn chain_txs(chain: &SparseChain) -> (Vec<(TxHeight, Txid)>, Vec<(TxHeight, Txid)>) {
    chain
        .iter_txids()
        .partition::<Vec<_>, _>(|(h, _)| h.is_confirmed())
}

/// TODO: Split into separate tests
#[test]
fn check_last_valid_rules() {
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(0, 0)])),
        Ok(ChangeSet {
            checkpoints: [(0, Change::new_insertion(gen_hash(0)))].into(),
            ..Default::default()
        }),
        "add first tip should succeed",
    );

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(0, 0), gen_block_id(1, 1)])),
        Ok(ChangeSet {
            checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
            ..Default::default()
        }),
        "applying second tip on top of first should succeed"
    );

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(2, 2)])),
        Err(UpdateFailure::NotConnected),
        "applying tip without specifying last valid should fail",
    );

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(1, 2), gen_block_id(3, 3,)])),
        Err(UpdateFailure::NotConnected),
        "apply tip, while specifying non-existant last_valid should fail",
    );

    assert_eq!(
        chain.apply_update(&gen_update([
            gen_block_id(0, 100),
            gen_block_id(1, 101),
            gen_block_id(10, 10)
        ])),
        Ok(ChangeSet {
            checkpoints: [
                (0, Change::new_alteration(gen_hash(0), gen_hash(100))),
                (1, Change::new_alteration(gen_hash(1), gen_hash(101))),
                (10, Change::new_insertion(gen_hash(10))),
            ]
            .into(),
            ..Default::default()
        }),
        "this update should empty the chain before introducing new checkpoints",
    );

    assert_eq!(
        chain.apply_update(&gen_update([
            gen_block_id(10, 10), // last valid
            gen_block_id(4, 4),
            gen_block_id(3, 3)
        ])),
        Ok(ChangeSet {
            checkpoints: [
                (3, Change::new_insertion(gen_hash(3))),
                (4, Change::new_insertion(gen_hash(4))),
            ]
            .into(),
            txids: [].into()
        }),
        "arbitary block can be introduced when last_valid exists"
    );

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(5, 5), gen_block_id(9, 9)])),
        Err(UpdateFailure::NotConnected),
        "arbitary blocks cannot be introduced with no connection",
    );

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(5, 5), gen_block_id(11, 11)])),
        Err(UpdateFailure::NotConnected),
        "arbitary blocks cannot be introduced with no connection (2)",
    );
}

/// Update and chain does not connect:
/// ```
///        | 0 | 1 | 2 | 3 | 4
/// chain  |     B   C
/// update | A   B       D
/// ```
/// This should fail as we cannot figure out whether C & D are on the same chain
#[test]
fn update_and_chain_does_not_connect() {
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_update(&gen_update([gen_block_id(1, 1), gen_block_id(2, 2)])),
        Ok(ChangeSet {
            checkpoints: [
                (1, Change::new_insertion(gen_hash(1))),
                (2, Change::new_insertion(gen_hash(2)))
            ]
            .into(),
            ..Default::default()
        }),
    );

    assert_eq!(
        chain.apply_update(&gen_update([
            gen_block_id(0, 0),
            gen_block_id(1, 1),
            gen_block_id(3, 3)
        ])),
        Err(UpdateFailure::NotConnected),
    );
}

#[test]
fn apply_tips() {
    let mut chain = SparseChain::default();

    // gen 10 checkpoints
    let mut last_valid = None;
    for i in 0..10 {
        let new_tip = gen_block_id(i, i as _);
        chain
            .apply_update(&gen_update(
                last_valid.iter().chain(core::iter::once(&new_tip)).cloned(),
            ))
            .expect("should succeed");
        last_valid = Some(new_tip);
    }

    // repeated last tip should succeed
    assert_eq!(
        chain
            .apply_update(&gen_update(last_valid))
            .expect("repeated last_tip should succeed"),
        ChangeSet::default(),
    );

    // ensure state of sparsechain is correct
    chain
        .range_checkpoints(..)
        .zip(0..)
        .for_each(|(block_id, exp_height)| {
            assert_eq!(block_id, gen_block_id(exp_height, exp_height as _))
        });
}

#[test]
fn checkpoint_limit_is_respected() {
    let mut chain = SparseChain::default();
    chain.set_checkpoint_limit(Some(5));

    // gen 10 checkpoints
    let mut last_valid = None;
    for i in 0..10 {
        let new_tip = gen_block_id(i, i as _);

        let changes = chain
            .apply_update(&Update {
                txs: [(gen_txid(i as _).into(), TxHeight::Confirmed(i))].into(),
                ..gen_update(last_valid.iter().chain(core::iter::once(&new_tip)).cloned())
            })
            .expect("should succeed");

        assert_eq!(
            changes,
            ChangeSet {
                // TODO: Figure out whether checkpoint pruning should be represented in changeset
                checkpoints: //if i < 5 {
                    [(i, Change::new_insertion(gen_hash(i as _)))].into(),
                // } else {
                //     [
                //         (i, Change::new_insertion(gen_hash(i as _))),
                //         (i - 5, Change::new_removal(gen_hash((i - 5) as _))),
                //     ]
                //     .into()
                // },
                txids: [(
                    gen_hash(i as _),
                    Change::new_insertion(TxHeight::Confirmed(i))
                )]
                .into(),
            }
        );

        last_valid = Some(new_tip);
    }

    assert_eq!(chain.iter_txids().count(), 10);
    assert_eq!(chain.range_checkpoints(..).count(), 5);
}

#[test]
fn add_txids() {
    let mut chain = SparseChain::default();

    let update = Update {
        txs: (0..100)
            .map(gen_hash::<Txid>)
            .map(|txid| (txid.into(), TxHeight::Confirmed(1)))
            .collect(),
        ..gen_update([gen_block_id(1, 1)])
    };

    assert_eq!(
        chain.apply_update(&update),
        Ok(ChangeSet {
            checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
            txids: update
                .txs
                .iter()
                .map(|(txid, height)| (txid.txid(), Change::new_insertion(*height)))
                .collect(),
        }),
        "add many txs in single checkpoint should succeed",
    );

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(2).into(), TxHeight::Confirmed(3))].into(),
                ..gen_update([gen_block_id(1, 1), gen_block_id(2, 2)])
            })
            .expect_err("update that adds tx with height greater than hew tip should fail"),
        UpdateFailure::Bogus(BogusReason::TxHeightGreaterThanTip {
            txid: gen_hash(2),
            tx_height: 3,
            tip_height: 2,
        })
    );
}

#[test]
fn add_txs_of_same_height_with_different_updates() {
    let mut chain = SparseChain::default();
    let block = gen_block_id(0, 0);

    // add one block
    assert_eq!(
        chain.apply_update(&gen_update([block])),
        Ok(ChangeSet {
            checkpoints: [(0, Change::new_insertion(gen_hash(0)))].into(),
            ..Default::default()
        }),
        "should succeed",
    );

    // add txs of same height with different updates
    (0..100).for_each(|i| {
        assert_eq!(
            chain
                .apply_update(&Update {
                    txs: [(gen_txid(i as _).into(), TxHeight::Confirmed(0))].into(),
                    ..gen_update([block])
                })
                .expect("should succeed"),
            ChangeSet {
                txids: [(
                    gen_hash(i as _),
                    Change::new_insertion(TxHeight::Confirmed(0))
                )]
                .into(),
                ..Default::default()
            }
        );
    });

    assert_eq!(chain.iter_txids().count(), 100);
    assert_eq!(chain.range_checkpoints(..).count(), 1);
}

#[test]
fn confirm_tx() {
    let mut chain = SparseChain::default();

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [
                    (gen_txid(10).into(), TxHeight::Unconfirmed),
                    (gen_txid(20).into(), TxHeight::Unconfirmed),
                ]
                .into(),
                ..gen_update([gen_block_id(1, 1)])
            })
            .expect("adding two txs from mempool should succeed"),
        ChangeSet {
            checkpoints: [(1, Change::new_insertion(gen_hash(1)))].into(),
            txids: [
                (gen_hash(10), Change::new_insertion(TxHeight::Unconfirmed)),
                (gen_hash(20), Change::new_insertion(TxHeight::Unconfirmed))
            ]
            .into()
        }
    );

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(10).into(), TxHeight::Confirmed(0))].into(),
                ..gen_update([gen_block_id(1, 1), gen_block_id(1, 1)])
            })
            .expect("it should be okay to confirm tx into block before last_valid (partial sync)"),
        ChangeSet {
            txids: [(
                gen_hash(10),
                Change::new(Some(TxHeight::Unconfirmed), Some(TxHeight::Confirmed(0)))
            )]
            .into(),
            ..Default::default()
        }
    );
    let (confirmed, unconfirmed) = chain_txs(&chain);
    assert_eq!(confirmed.len(), 1);
    assert_eq!(unconfirmed.len(), 1);

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(20).into(), TxHeight::Confirmed(2))].into(),
                ..gen_update([gen_block_id(1, 1), gen_block_id(2, 2)])
            })
            .expect("it should be okay to confirm tx into the tip introduced"),
        ChangeSet {
            checkpoints: [(2, Change::new_insertion(gen_hash(2)))].into(),
            txids: [(
                gen_hash(20),
                Change::new(Some(TxHeight::Unconfirmed), Some(TxHeight::Confirmed(2)))
            )]
            .into(),
        }
    );
    let (confirmed, unconfirmed) = chain_txs(&chain);
    assert_eq!(confirmed.len(), 2);
    assert_eq!(unconfirmed.len(), 0);

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(10).into(), TxHeight::Unconfirmed)].into(),
                ..gen_update([gen_block_id(2, 2), gen_block_id(2, 2)])
            })
            .expect_err("tx cannot be unconfirmed without invalidate"),
        UpdateFailure::Inconsistent {
            inconsistent_txid: gen_hash(10),
            original_height: TxHeight::Confirmed(0),
            update_height: TxHeight::Unconfirmed,
        }
    );

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(20).into(), TxHeight::Confirmed(3))].into(),
                ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
            })
            .expect_err("tx cannot move forward in blocks without invalidate"),
        UpdateFailure::Inconsistent {
            inconsistent_txid: gen_hash(20),
            original_height: TxHeight::Confirmed(2),
            update_height: TxHeight::Confirmed(3),
        },
    );

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(20).into(), TxHeight::Confirmed(1))].into(),
                ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
            })
            .expect_err("tx cannot move backwards in blocks without invalidate"),
        UpdateFailure::Inconsistent {
            inconsistent_txid: gen_hash(20),
            original_height: TxHeight::Confirmed(2),
            update_height: TxHeight::Confirmed(1),
        },
    );

    assert_eq!(
        chain
            .apply_update(&Update {
                txs: [(gen_txid(20).into(), TxHeight::Confirmed(2))].into(),
                ..gen_update([gen_block_id(2, 2), gen_block_id(3, 3)])
            })
            .expect("update can introduce already-existing tx"),
        ChangeSet {
            checkpoints: [(3, Change::new_insertion(gen_hash(3)))].into(),
            ..Default::default()
        }
    );
    let (confirmed, unconfirmed) = chain_txs(&chain);
    assert_eq!(confirmed.len(), 2);
    assert_eq!(unconfirmed.len(), 0);
}
