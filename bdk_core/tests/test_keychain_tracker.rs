use bdk_core::*;
mod checkpoint_gen;
use bitcoin::OutPoint;
use checkpoint_gen::{CheckpointGen, ISpec, OSpec, TxSpec};

#[test]
fn no_checkpoint_and_then_confirm() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 0);

    let mut checkpoint = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: None,
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(checkpoint.clone()), ApplyResult::Ok);

    assert_eq!(
        chain.iter_checkpoints(..).count(),
        0,
        "adding tx to mempool doesn't create checkpoint"
    );
    assert_eq!(chain.iter_tx().count(), 1);
    tracker.sync(&chain);

    {
        let txouts = tracker.iter_txout_full(&chain).collect::<Vec<_>>();
        let unspent = tracker.iter_unspent_full(&chain).collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.value, 1_000);
        assert_eq!(txout.confirmed_at, None);
        assert_eq!(unspent, txouts);
    }

    checkpoint.transactions[0].confirmation_time = Some(BlockTime { height: 1, time: 1 });
    checkpoint.new_tip.height += 1;

    assert_eq!(chain.apply_checkpoint(checkpoint), ApplyResult::Ok);
    {
        let txouts = tracker.iter_txout_full(&chain).collect::<Vec<_>>();
        let unspent = tracker.iter_unspent_full(&chain).collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.value, 1_000);
        assert_eq!(txout.confirmed_at, Some(BlockTime { height: 1, time: 1 }));
        assert_eq!(unspent, txouts);
    }
}

#[test]
fn orphaned_txout_no_longer_appears() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let checkpoint1 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(checkpoint1.clone()), ApplyResult::Ok);
    tracker.sync(&chain);

    let mut checkpoint2 = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_001, 1), OSpec::Other(1_800)],
            confirmed_at: Some(0),
        }],
        0,
    );

    checkpoint2.invalidate = Some(checkpoint1.new_tip);
    assert_eq!(chain.apply_checkpoint(checkpoint2.clone()), ApplyResult::Ok);
    tracker.sync(&chain);

    assert_eq!(chain.apply_checkpoint(checkpoint2), ApplyResult::Ok);
    {
        let txouts = tracker.iter_txout_full(&chain).collect::<Vec<_>>();
        let unspent = tracker.iter_unspent_full(&chain).collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.value, 1_001);
        assert_eq!(txout.confirmed_at, Some(BlockTime { height: 0, time: 0 }));
        assert_eq!(unspent, txouts);
    }
}

#[test]
fn output_spend_and_created_in_same_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let checkpoint1 = checkpoint_gen.create_update(
        vec![
            TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
                confirmed_at: Some(0),
            },
            TxSpec {
                inputs: vec![ISpec::Other, ISpec::InCheckPoint(0, 0)],
                outputs: vec![OSpec::Mine(3_000, 1)],
                confirmed_at: Some(0),
            },
        ],
        0,
    );

    assert_eq!(chain.apply_checkpoint(checkpoint1.clone()), ApplyResult::Ok);
    tracker.sync(&chain);

    {
        let txouts = tracker.iter_txout_full(&chain).collect::<Vec<_>>();
        let unspent = tracker.iter_unspent_full(&chain).collect::<Vec<_>>();
        assert_eq!(txouts.len(), 2);
        assert_eq!(unspent.len(), 1);

        assert!(txouts
            .iter()
            .find(|(_, txout)| txout.value == 1_000)
            .is_some());
        assert!(txouts
            .iter()
            .find(|(_, txout)| txout.value == 3_000)
            .is_some());
        assert_eq!(unspent[0].1.value, 3_000);
    }
}

#[test]
fn spend_unspent_in_reorg() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let first = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: Some(0),
        }],
        0,
    );

    let second = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 1), OSpec::Other(500)],
            confirmed_at: Some(1),
        }],
        1,
    );

    let mut third = checkpoint_gen.create_update(
        vec![TxSpec {
            inputs: vec![ISpec::Explicit(OutPoint {
                txid: first.transactions[0].tx.txid(),
                vout: 0,
            })],
            outputs: vec![OSpec::Other(500)],
            confirmed_at: Some(1),
        }],
        2,
    );

    third.invalidate = Some(second.new_tip);
    third.base_tip = Some(first.new_tip);

    assert_eq!(chain.apply_checkpoint(first), ApplyResult::Ok);
    tracker.sync(&chain);
    assert_eq!(tracker.iter_unspent(&chain).count(), 1);
    assert_eq!(tracker.iter_txout(&chain).count(), 1);

    assert_eq!(chain.apply_checkpoint(second), ApplyResult::Ok);
    tracker.sync(&chain);
    assert_eq!(tracker.iter_unspent(&chain).count(), 2);
    assert_eq!(tracker.iter_txout(&chain).count(), 2);

    assert_eq!(chain.apply_checkpoint(third), ApplyResult::Ok);
    tracker.sync(&chain);
    assert_eq!(tracker.iter_unspent(&chain).count(), 0);
    assert_eq!(tracker.iter_txout(&chain).count(), 1);
}
