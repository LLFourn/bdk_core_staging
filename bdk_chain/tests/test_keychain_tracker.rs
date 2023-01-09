#![cfg(feature = "miniscript")]
use bdk_chain::{
    keychain::KeychainTracker,
    miniscript::{
        bitcoin::{secp256k1::Secp256k1, OutPoint, PackedLockTime, Transaction, TxOut},
        Descriptor,
    },
    ConfirmationTime,
};

#[test]
fn test_insert_tx() {
    let mut tracker = KeychainTracker::default();
    let secp = Secp256k1::new();
    let (descriptor, _) = Descriptor::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
    tracker.add_keychain((), descriptor.clone());
    let txout = TxOut {
        value: 100_000,
        script_pubkey: descriptor.at_derivation_index(5).script_pubkey(),
    };

    let tx = Transaction {
        version: 0x01,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![txout],
    };

    assert!(tracker.txout_index.store_up_to(&(), 5));
    tracker
        .insert_tx(tx.clone(), Some(ConfirmationTime::Unconfirmed))
        .unwrap();
    assert_eq!(
        tracker
            .chain_graph()
            .transactions_in_chain()
            .collect::<Vec<_>>(),
        vec![(&ConfirmationTime::Unconfirmed, &tx)]
    );

    assert_eq!(
        tracker.txout_index.keychain_txouts(&()).collect::<Vec<_>>(),
        vec![(
            5,
            OutPoint {
                txid: tx.txid(),
                vout: 0
            }
        )]
    );
}
