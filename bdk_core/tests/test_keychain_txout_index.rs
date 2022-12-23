use bdk_core::collections::BTreeMap;

use bdk_core::keychain_txout_index::KeychainTxOutIndex;
use miniscript::{Descriptor, DescriptorPublicKey};

#[test]
fn test_store_all_up_to() {
    #[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
    enum MyKeychain {
        External,
        Internal,
    }

    let mut txout_index = KeychainTxOutIndex::<MyKeychain>::default();

    let secp = bdk_core::bitcoin::secp256k1::Secp256k1::signing_only();
    let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
    let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();

    txout_index.add_keychain(MyKeychain::External, external_descriptor);
    txout_index.add_keychain(MyKeychain::Internal, internal_descriptor);

    let derive_to: BTreeMap<_, _> = [(MyKeychain::External, 12), (MyKeychain::Internal, 24)].into();
    assert!(txout_index.store_all_up_to(&derive_to));
    assert_eq!(txout_index.derivation_indices(), derive_to);
}
