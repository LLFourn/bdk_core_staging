#![cfg(feature = "miniscript")]

#[macro_use]
mod common;
use bdk_chain::{
    collections::BTreeMap,
    keychain::{DerivationAdditions, KeychainTxOutIndex},
};
use bitcoin::{OutPoint, TxOut};

use miniscript::{Descriptor, DescriptorPublicKey};

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum TestKeychain {
    External,
    Internal,
}

fn init_txout_index() -> (
    bdk_chain::keychain::KeychainTxOutIndex<TestKeychain>,
    Descriptor<DescriptorPublicKey>,
    Descriptor<DescriptorPublicKey>,
) {
    let mut txout_index = bdk_chain::keychain::KeychainTxOutIndex::<TestKeychain>::default();

    let secp = bdk_chain::bitcoin::secp256k1::Secp256k1::signing_only();
    let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
    let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();

    txout_index.add_keychain(TestKeychain::External, external_descriptor.clone());
    txout_index.add_keychain(TestKeychain::Internal, internal_descriptor.clone());

    (txout_index, external_descriptor, internal_descriptor)
}

#[test]
fn test_store_all_up_to() {
    let (mut txout_index, _, _) = init_txout_index();
    let derive_to: BTreeMap<_, _> =
        [(TestKeychain::External, 12), (TestKeychain::Internal, 24)].into();
    assert_eq!(
        txout_index.store_all_up_to(&derive_to).as_inner(),
        &derive_to
    );
    assert_eq!(txout_index.derivation_indices(), derive_to);
    assert_eq!(
        txout_index.store_all_up_to(&derive_to),
        DerivationAdditions::default(),
        "no changes if we set to the same thing"
    );
}

#[test]
fn test_pad_all_with_unused() {
    let (mut txout_index, external_desc, _) = init_txout_index();

    let external_spk3 = external_desc.at_derivation_index(3).script_pubkey();

    assert_eq!(
        txout_index
            .store_up_to(&TestKeychain::External, 3)
            .as_inner(),
        &[(TestKeychain::External, 3)].into(),
    );
    txout_index.scan_txout(
        OutPoint::default(),
        &TxOut {
            value: 420,
            script_pubkey: external_spk3,
        },
    );

    assert_eq!(
        txout_index.pad_all_with_unused(5).as_inner(),
        &[(TestKeychain::External, 8), (TestKeychain::Internal, 4)].into(),
    );
    assert_eq!(
        txout_index.derivation_indices(),
        [(TestKeychain::External, 8), (TestKeychain::Internal, 4)].into()
    );
}

#[test]
fn test_wildcard_derivations() {
    let (mut txout_index, external_desc, _) = init_txout_index();
    let external_spk_0 = external_desc.at_derivation_index(0).script_pubkey();
    let external_spk_16 = external_desc.at_derivation_index(16).script_pubkey();
    let external_spk_26 = external_desc.at_derivation_index(26).script_pubkey();
    let external_spk_27 = external_desc.at_derivation_index(27).script_pubkey();

    // - nothing is derived
    // - unused list is also empty
    //
    // - next_derivation_index() == (0, true)
    // - derive_new() == ((0, <spk>), DerivationAdditions)
    // - next_unused() == ((0, <spk>), DerivationAdditions:is_empty())
    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (0, true)
    );
    let (spk, changeset) = txout_index.derive_new(&TestKeychain::External);
    assert_eq!(spk, (0_u32, &external_spk_0));
    assert_eq!(changeset.as_inner(), &[(TestKeychain::External, 0)].into());
    let (spk, changeset) = txout_index.next_unused(&TestKeychain::External);
    assert_eq!(spk, (0_u32, &external_spk_0));
    assert_eq!(changeset.as_inner(), &[].into());

    // - derived till 25
    // - used all spks till 15.
    // - used list : [0..=15, 17, 20, 23]
    // - unused list: [16, 18, 19, 21, 22, 24, 25]

    // - next_derivation_index() = (26, true)
    // - derive_new() = ((26, <spk>), DerivationAdditions)
    // - next_unused() == ((16, <spk>), DerivationAdditions::is_empty())
    let _ = txout_index.store_up_to(&TestKeychain::External, 25);

    (0..=15)
        .into_iter()
        .chain([17, 20, 23].into_iter())
        .for_each(|index| txout_index.mark_used(&TestKeychain::External, index));

    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (26, true)
    );

    let (spk, changeset) = txout_index.derive_new(&TestKeychain::External);
    assert_eq!(spk, (26, &external_spk_26));

    assert_eq!(changeset.as_inner(), &[(TestKeychain::External, 26)].into());

    let (spk, changeset) = txout_index.next_unused(&TestKeychain::External);
    assert_eq!(spk, (16, &external_spk_16));
    assert_eq!(changeset.as_inner(), &[].into());

    // - Use all the derived till 26.
    // - next_unused() = ((27, <spk>), DerivationAdditions)
    (0..=26)
        .into_iter()
        .for_each(|index| txout_index.mark_used(&TestKeychain::External, index));

    let (spk, changeset) = txout_index.next_unused(&TestKeychain::External);
    assert_eq!(spk, (27, &external_spk_27));
    assert_eq!(changeset.as_inner(), &[(TestKeychain::External, 27)].into());
}

#[test]
fn test_non_wildcard_derivations() {
    let mut txout_index = KeychainTxOutIndex::<TestKeychain>::default();

    let secp = bitcoin::secp256k1::Secp256k1::signing_only();
    let (no_wildcard_descriptor, _) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "wpkh([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/0)").unwrap();
    let external_spk = no_wildcard_descriptor
        .at_derivation_index(0)
        .script_pubkey();

    txout_index.add_keychain(TestKeychain::External, no_wildcard_descriptor);

    // given:
    // - `txout_index` with no stored scripts
    // expect:
    // - next derivation index should be new
    // - when we derive a new script, script @ index 0
    // - when we get the next unused script, script @ index 0
    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (0, true)
    );
    let (spk, changeset) = txout_index.derive_new(&TestKeychain::External);
    assert_eq!(spk, (0, &external_spk));
    assert_eq!(changeset.as_inner(), &[(TestKeychain::External, 0)].into());

    let (spk, changeset) = txout_index.next_unused(&TestKeychain::External);
    assert_eq!(spk, (0, &external_spk));
    assert_eq!(changeset.as_inner(), &[].into());

    // given:
    // - the non-wildcard descriptor already has a stored and used script
    // expect:
    // - next derivation index should not be new
    // - derive new and next unused should return the old script
    // - store_up_to should not panic and return empty additions
    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (0, false)
    );
    txout_index.mark_used(&TestKeychain::External, 0);

    let (spk, changeset) = txout_index.derive_new(&TestKeychain::External);
    assert_eq!(spk, (0, &external_spk));
    assert_eq!(changeset.as_inner(), &[].into());

    let (spk, changeset) = txout_index.next_unused(&TestKeychain::External);
    assert_eq!(spk, (0, &external_spk));
    assert_eq!(changeset.as_inner(), &[].into());
    assert!(txout_index
        .store_up_to(&TestKeychain::External, 200)
        .is_empty());
}
