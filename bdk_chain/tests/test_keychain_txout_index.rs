#![cfg(feature = "miniscript")]
use bdk_chain::collections::BTreeMap;

use bdk_chain::keychain::KeychainTxOutIndex;
use bitcoin::{OutPoint, TxOut};
use miniscript::{Descriptor, DescriptorPublicKey};

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum TestKeychain {
    External,
    Internal,
}

fn init_txout_index() -> KeychainTxOutIndex<TestKeychain> {
    let mut txout_index = KeychainTxOutIndex::<TestKeychain>::default();

    let secp = bdk_chain::bitcoin::secp256k1::Secp256k1::signing_only();
    let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
    let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();

    txout_index.add_keychain(TestKeychain::External, external_descriptor);
    txout_index.add_keychain(TestKeychain::Internal, internal_descriptor);

    txout_index
}

#[test]
fn test_store_all_up_to() {
    let mut txout_index = init_txout_index();
    let derive_to: BTreeMap<_, _> =
        [(TestKeychain::External, 12), (TestKeychain::Internal, 24)].into();
    assert!(txout_index.store_all_up_to(&derive_to));
    assert_eq!(txout_index.derivation_indices(), derive_to);
}

#[test]
fn test_pad_all_with_unused() {
    let mut txout_index = init_txout_index();

    let external_spk3 = txout_index
        .keychains()
        .get(&TestKeychain::External)
        .unwrap()
        .at_derivation_index(3)
        .script_pubkey();

    assert!(txout_index.store_up_to(&TestKeychain::External, 3));
    txout_index.scan_txout(
        OutPoint::default(),
        &TxOut {
            value: 420,
            script_pubkey: external_spk3,
        },
    );

    assert!(txout_index.pad_all_with_unused(5));
    assert_eq!(
        txout_index.derivation_indices(),
        [(TestKeychain::External, 8), (TestKeychain::Internal, 4)].into()
    );
}
