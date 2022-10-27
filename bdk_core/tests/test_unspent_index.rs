use bdk_core::{
    testing::{gen_block_id, gen_hash},
    BlockId, Change, ChangeSet, SparseChain, SpkTracker, TxGraph, TxHeight, Unspent, UnspentIndex,
    Update,
};
use bitcoin::{
    OutPoint, PackedLockTime, Script, Sequence, Transaction, TxIn, TxOut, Txid, Witness,
};

#[test]
fn single_spends() {
    let mut chain = SparseChain::default();
    let mut graph = TxGraph::default();
    let mut tracker = SpkTracker::<()>::default();
    let mut index = UnspentIndex::<()>::default();

    let spk = Script::new_p2pkh(&gen_hash(2000));
    tracker.add_spk((), spk.clone());

    let value = 50_000;

    let mut prev_txid = Option::<Txid>::None;
    let mut prev_last_valid = Option::<BlockId>::None;

    for i in 0..100 {
        let tx = Transaction {
            version: 0,
            lock_time: PackedLockTime::ZERO,
            input: prev_txid
                .iter()
                .map(|&txid| TxIn {
                    previous_output: OutPoint::new(txid, 0),
                    script_sig: Script::new(),
                    sequence: Sequence::default(),
                    witness: Witness::new(),
                })
                .collect(),
            output: vec![TxOut {
                script_pubkey: spk.clone(),
                value,
            }],
        };
        prev_txid.replace(tx.txid());
        assert!(graph.insert_tx(&tx), "graph should insert new tx");

        let new_tip = gen_block_id(i, i as _);
        let update = Update {
            txids: [(tx.txid(), TxHeight::Confirmed(i))].into(),
            ..Update::new(prev_last_valid.replace(new_tip), new_tip)
        };

        let change_set = chain.apply_update(update).expect("update should succeed");

        assert_eq!(
            change_set,
            ChangeSet {
                checkpoints: [(i, Change::new_insertion(new_tip.hash))].into(),
                txids: [(tx.txid(), Change::new_insertion(TxHeight::Confirmed(i)))].into(),
            }
        );

        let op = OutPoint::new(tx.txid(), 0);

        tracker
            .sync(&graph, &change_set)
            .expect("sync spk tracker should succeed");

        assert_eq!(tracker.iter_txout().len(), (i + 1) as _);
        assert_eq!(tracker.txout(op), Some(((), &tx.output[0])));

        index
            .sync(&chain, &graph, &tracker)
            .expect("sync unspent index should succeed");

        assert_eq!(index.iter().len(), 1, "failed in round {}", i);
        assert_eq!(
            index.unspent(op),
            Some(Unspent {
                outpoint: op,
                txout: TxOut {
                    script_pubkey: spk.clone(),
                    value
                },
                spk_index: (),
                height: TxHeight::Confirmed(i)
            })
        );
    }
}
