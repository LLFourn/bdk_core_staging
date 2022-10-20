use bdk_core::*;
use bitcoin::{hashes::Hash, Txid};

fn gen_hash<H: Hash>(n: u64) -> H {
    let data = n.to_le_bytes();
    Hash::hash(&data[..])
}

fn gen_block_id(height: u32, hash_n: u64) -> BlockId {
    BlockId {
        height,
        hash: gen_hash(hash_n),
    }
}

#[test]
fn check_last_valid_rules() {
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(None, gen_block_id(0, 0))),
        ApplyResult::Ok,
        "add first tip should succeed",
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(0, 0)),
            gen_block_id(1, 1)
        )),
        ApplyResult::Ok,
        "applying second tip on top of first should succeed",
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(None, gen_block_id(2, 2))),
        ApplyResult::Stale(StaleReason::UnexpectedLastValid {
            got: None,
            expected: Some(gen_block_id(1, 1))
        }),
        "applying third tip on top without specifying last valid should fail",
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(1, 2)),
            gen_block_id(3, 3),
        )),
        ApplyResult::Stale(StaleReason::UnexpectedLastValid {
            got: Some(gen_block_id(1, 2)),
            expected: Some(gen_block_id(1, 1)),
        }),
        "applying new tip, in which suppled last_valid is non-existant, should fail",
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(1, 1)),
            gen_block_id(1, 3),
        )),
        ApplyResult::Stale(StaleReason::LastValidConflictsNewTip {
            last_valid: gen_block_id(1, 1),
            new_tip: gen_block_id(1, 3),
        }),
        "applying new tip, in which new_tip conflicts last_valid, should fail",
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(1, 1)),
            gen_block_id(0, 3),
        )),
        ApplyResult::Stale(StaleReason::LastValidConflictsNewTip {
            last_valid: gen_block_id(1, 1),
            new_tip: gen_block_id(0, 3),
        }),
        "applying new tip, in which new_tip conflicts last_valid, should fail (2)",
    );
}

#[test]
fn check_invalidate_rules() {
    let mut chain = SparseChain::default();

    // add one checkpoint
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(None, gen_block_id(1, 1))),
        ApplyResult::Ok
    );

    // when we are invalidating the one and only checkpoint, `last_valid` should be `None`
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            invalidate: Some(gen_block_id(1, 1)),
            ..CheckpointCandidate::new(Some(gen_block_id(1, 1)), gen_block_id(1, 2))
        }),
        ApplyResult::Stale(StaleReason::UnexpectedLastValid {
            got: Some(gen_block_id(1, 1)),
            expected: None,
        }),
        "should fail when invalidate does not directly preceed last_valid",
    );
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            invalidate: Some(gen_block_id(1, 1)),
            ..CheckpointCandidate::new(None, gen_block_id(1, 2))
        }),
        ApplyResult::Ok,
        "invalidate should succeed",
    );

    // add two checkpoints
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(1, 2)),
            gen_block_id(2, 3)
        )),
        ApplyResult::Ok
    );
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(
            Some(gen_block_id(2, 3)),
            gen_block_id(3, 4),
        )),
        ApplyResult::Ok
    );

    // `invalidate` should directly follow `last_valid`
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            invalidate: Some(gen_block_id(3, 4)),
            ..CheckpointCandidate::new(Some(gen_block_id(1, 2)), gen_block_id(3, 5))
        }),
        ApplyResult::Stale(StaleReason::UnexpectedLastValid {
            got: Some(gen_block_id(1, 2)),
            expected: Some(gen_block_id(2, 3)),
        }),
        "should fail when checkpoint directly following last_valid is not invalidate",
    );
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            invalidate: Some(gen_block_id(3, 4)),
            ..CheckpointCandidate::new(Some(gen_block_id(2, 3)), gen_block_id(3, 5))
        }),
        ApplyResult::Ok,
        "should succeed",
    );
}

#[test]
fn apply_tips() {
    let mut chain = SparseChain::default();

    // gen 10 checkpoints
    let mut last_valid = None;
    for i in 0..10 {
        let new_tip = gen_block_id(i, i as _);
        assert_eq!(
            chain.apply_checkpoint(CheckpointCandidate::new(last_valid, new_tip)),
            ApplyResult::Ok,
        );
        last_valid = Some(new_tip);
    }

    // repeated last tip should succeed
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(last_valid, last_valid.unwrap())),
        ApplyResult::Ok,
        "repeated last_tip should succeed"
    );

    // ensure state of sparsechain is correct
    chain
        .iter_checkpoints(..)
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
        assert_eq!(
            chain.apply_checkpoint(CheckpointCandidate {
                txids: vec![(gen_hash(i as _), Some(i)),],
                ..CheckpointCandidate::new(last_valid, new_tip)
            }),
            ApplyResult::Ok,
        );
        last_valid = Some(new_tip);
    }

    assert_eq!(chain.iter_confirmed_txids().count(), 10);
    let latest = chain.latest_checkpoint();
    assert_eq!(chain.iter_checkpoints(..).count(), 5);
    assert_eq!(chain.latest_checkpoint(), latest);
}

#[test]
fn add_txids() {
    let mut chain = SparseChain::default();

    let txids_1 = (0..100)
        .map(gen_hash::<Txid>)
        .map(|txid| (txid, Some(1)))
        .collect();

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: txids_1,
            ..CheckpointCandidate::new(None, gen_block_id(1, 1))
        }),
        ApplyResult::Ok,
        "add many txs in single checkpoint should succeed"
    );

    // TODO:
    // * Check txs in sparsechain
    // * Check getting height of txids
    // * Check getting txid of height
    // * Transaction height of tx should change when new checkpoint is added

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(2), Some(3))],
            ..CheckpointCandidate::new(Some(gen_block_id(1, 1)), gen_block_id(2, 2))
        }),
        ApplyResult::Stale(StaleReason::TxidHeightGreaterThanTip {
            new_tip: gen_block_id(2, 2),
            txid: (gen_hash(2), Some(3)),
        }),
        "adding tx with height greater than new tip should fail",
    );
}

#[test]
fn add_txs_of_same_height_with_different_updates() {
    let mut chain = SparseChain::default();
    let block = gen_block_id(0, 0);

    // add one block
    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate::new(None, block)),
        ApplyResult::Ok
    );

    // add txs of same height with different updates
    (0..100).for_each(|i| {
        assert_eq!(
            chain.apply_checkpoint(CheckpointCandidate {
                txids: vec![(gen_hash(i as _), Some(0))],
                ..CheckpointCandidate::new(Some(block), block)
            }),
            ApplyResult::Ok,
        );
    });

    assert_eq!(chain.iter_txids().count(), 100);
    assert_eq!(chain.iter_confirmed_txids().count(), 100);
    assert_eq!(chain.iter_mempool_txids().count(), 0);
    assert_eq!(chain.iter_checkpoints(..).count(), 1);
}

#[test]
fn confirm_tx() {
    let mut chain = SparseChain::default();

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(10), None), (gen_hash(20), None)],
            ..CheckpointCandidate::new(None, gen_block_id(1, 1))
        }),
        ApplyResult::Ok,
        "adding two txs from mempool should succeed"
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(10), Some(0))],
            ..CheckpointCandidate::new(Some(gen_block_id(1, 1)), gen_block_id(1, 1))
        }),
        ApplyResult::Ok,
        "it should be okay to confirm tx into block before last_valid (partial sync)",
    );
    assert_eq!(chain.iter_txids().count(), 2);
    assert_eq!(chain.iter_confirmed_txids().count(), 1);
    assert_eq!(chain.iter_mempool_txids().count(), 1);

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(20), Some(2))],
            ..CheckpointCandidate::new(Some(gen_block_id(1, 1)), gen_block_id(2, 2))
        }),
        ApplyResult::Ok,
        "it should be okay to confirm tx into the tip introduced",
    );
    assert_eq!(chain.iter_txids().count(), 2);
    assert_eq!(chain.iter_confirmed_txids().count(), 2);
    assert_eq!(chain.iter_mempool_txids().count(), 0);

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(10), None)],
            ..CheckpointCandidate::new(Some(gen_block_id(2, 2)), gen_block_id(2, 2))
        }),
        ApplyResult::Stale(StaleReason::TxUnexpectedlyMoved {
            txid: gen_hash(10),
            from: Some(0),
            to: None,
        }),
        "tx cannot be unconfirmed without invalidate"
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(20), Some(3))],
            ..CheckpointCandidate::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
        }),
        ApplyResult::Stale(StaleReason::TxUnexpectedlyMoved {
            txid: gen_hash(20),
            from: Some(2),
            to: Some(3),
        }),
        "tx cannot move forward in blocks without invalidate"
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(20), Some(1))],
            ..CheckpointCandidate::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
        }),
        ApplyResult::Stale(StaleReason::TxUnexpectedlyMoved {
            txid: gen_hash(20),
            from: Some(2),
            to: Some(1),
        }),
        "tx cannot move backwards in blocks without invalidate"
    );

    assert_eq!(
        chain.apply_checkpoint(CheckpointCandidate {
            txids: vec![(gen_hash(20), Some(2))],
            ..CheckpointCandidate::new(Some(gen_block_id(2, 2)), gen_block_id(3, 3))
        }),
        ApplyResult::Ok,
        "update can introduce already-existing tx"
    );
    assert_eq!(chain.iter_txids().count(), 2);
    assert_eq!(chain.iter_confirmed_txids().count(), 2);
    assert_eq!(chain.iter_mempool_txids().count(), 0);
}

// TODO: Implement consistency detection
// TODO: add test for adding the target
