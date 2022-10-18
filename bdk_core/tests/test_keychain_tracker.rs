use bdk_core::*;
mod checkpoint_gen;
use bitcoin::OutPoint;
use checkpoint_gen::{CheckpointGen, ISpec, OSpec, TxSpec};

#[test]
fn add_single_unconfirmed_tx_and_then_confirm_it() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 0);

    let mut checkpoint = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: None,
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(checkpoint.clone()), ApplyResult::Ok);

    assert_eq!(chain.iter_checkpoints(..).count(), 1);
    assert_eq!(
        chain
            .checkpoint_txids(chain.latest_checkpoint().unwrap())
            .count(),
        0,
        "the checkpoint should be empty because tx is not confirmed"
    );

    assert_eq!(chain.iter_txids().count(), 1);
    tracker.sync(&graph);

    {
        let txouts = tracker.iter_txout_full(&chain, &graph).collect::<Vec<_>>();
        let unspent = tracker
            .iter_unspent_full(&chain, &graph)
            .collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.txout.value, 1_000);
        assert_eq!(txout.height, TxHeight::Unconfirmed);
        assert_eq!(unspent, txouts);
    }

    checkpoint.txids[0].1 = Some(1);
    checkpoint.new_tip.height += 1;

    assert_eq!(chain.apply_checkpoint(checkpoint), ApplyResult::Ok);

    {
        assert_eq!(chain.iter_checkpoints(..).count(), 2);
        assert_eq!(chain.latest_checkpoint().unwrap().height, 1);
        let txouts = tracker.iter_txout_full(&chain, &graph).collect::<Vec<_>>();
        let unspent = tracker
            .iter_unspent_full(&chain, &graph)
            .collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.txout.value, 1_000);
        assert_eq!(txout.height, TxHeight::Confirmed(1));
        assert_eq!(unspent, txouts);
    }
}

#[test]
fn orphaned_txout_no_longer_appears() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let checkpoint1 = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: Some(0),
        }],
        0,
    );

    assert_eq!(chain.apply_checkpoint(checkpoint1.clone()), ApplyResult::Ok);
    tracker.sync(&graph);

    let checkpoint2 = CheckpointCandidate {
        last_valid: None,
        invalidate: Some(checkpoint1.new_tip),
        ..checkpoint_gen.create_update(
            &mut graph,
            vec![TxSpec {
                inputs: vec![ISpec::Other],
                outputs: vec![OSpec::Mine(1_001, 1), OSpec::Other(1_800)],
                confirmed_at: Some(0),
            }],
            0,
        )
    };

    assert_eq!(chain.apply_checkpoint(checkpoint2.clone()), ApplyResult::Ok);
    tracker.sync(&graph);

    assert_eq!(chain.apply_checkpoint(checkpoint2), ApplyResult::Ok);
    {
        let txouts = tracker.iter_txout_full(&chain, &graph).collect::<Vec<_>>();
        let unspent = tracker
            .iter_unspent_full(&chain, &graph)
            .collect::<Vec<_>>();
        assert_eq!(txouts.len(), 1);
        let (spk_index, txout) = txouts[0].clone();
        assert_eq!(spk_index, ((), 0));
        assert_eq!(txout.txout.value, 1_001);
        assert_eq!(txout.height, TxHeight::Confirmed(0));
        assert_eq!(unspent, txouts);
    }
}

#[test]
fn output_spend_and_created_in_same_checkpoint() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let checkpoint1 = checkpoint_gen.create_update(
        &mut graph,
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
    tracker.sync(&graph);

    {
        let txouts = tracker.iter_txout_full(&chain, &graph).collect::<Vec<_>>();
        let unspent = tracker
            .iter_unspent_full(&chain, &graph)
            .collect::<Vec<_>>();
        assert_eq!(txouts.len(), 2);
        assert_eq!(unspent.len(), 1);

        assert!(txouts
            .iter()
            .find(|(_, txout)| txout.txout.value == 1_000)
            .is_some());
        assert!(txouts
            .iter()
            .find(|(_, txout)| txout.txout.value == 3_000)
            .is_some());
        assert_eq!(unspent[0].1.txout.value, 3_000);
    }
}

#[test]
fn spend_unspent_in_reorg() {
    let mut checkpoint_gen = CheckpointGen::new();
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    let mut tracker = KeychainTracker::default();
    tracker.add_keychain((), checkpoint_gen.descriptor.clone());
    tracker.derive_spks((), 2);

    let first = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 0), OSpec::Other(500)],
            confirmed_at: None,
        }],
        0,
    );
    assert_eq!(chain.apply_checkpoint(first.clone()), ApplyResult::Ok);
    tracker.sync(&graph);
    assert_eq!(tracker.iter_unspent(&chain, &graph).count(), 1); // TODO: fails here
    assert_eq!(tracker.iter_txout().count(), 1);

    let second = checkpoint_gen.create_update(
        &mut graph,
        vec![TxSpec {
            inputs: vec![ISpec::Other],
            outputs: vec![OSpec::Mine(1_000, 1), OSpec::Other(500)],
            confirmed_at: Some(1),
        }],
        1,
    );
    assert_eq!(chain.apply_checkpoint(second.clone()), ApplyResult::Ok);
    tracker.sync(&graph);
    assert_eq!(tracker.iter_unspent(&chain, &graph).count(), 2);
    assert_eq!(tracker.iter_txout().count(), 2);

    let third = CheckpointCandidate {
        last_valid: Some(first.new_tip),
        invalidate: Some(second.new_tip),
        ..checkpoint_gen.create_update(
            &mut graph,
            vec![TxSpec {
                inputs: vec![ISpec::Explicit(OutPoint {
                    txid: first.txids[0].0,
                    vout: 0,
                })],
                outputs: vec![OSpec::Other(500)],
                confirmed_at: Some(1),
            }],
            2,
        )
    };
    assert_eq!(chain.apply_checkpoint(third), ApplyResult::Ok);
    tracker.sync(&graph);
    assert_eq!(tracker.iter_unspent(&chain, &graph).count(), 0);
    assert_eq!(tracker.iter_txout().count(), 2);
}
