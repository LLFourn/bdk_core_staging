#![cfg(feature = "miniscript")]

#[macro_use]
mod common;
use bdk_chain::{collections::BTreeMap, keychain::KeychainTxOutIndex};
use bitcoin::{hashes::hex::FromHex, OutPoint, Script, TxOut};

use miniscript::{Descriptor, DescriptorPublicKey};

#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum TestKeychain {
    External,
    Internal,
}

fn init_txout_index() -> bdk_chain::keychain::KeychainTxOutIndex<TestKeychain> {
    let mut txout_index = bdk_chain::keychain::KeychainTxOutIndex::<TestKeychain>::default();

    let secp = bdk_chain::bitcoin::secp256k1::Secp256k1::signing_only();
    let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
    let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();

    txout_index.add_keychain(TestKeychain::External, external_descriptor);
    txout_index.add_keychain(TestKeychain::Internal, internal_descriptor);

    txout_index
}

fn mark_used(tracker: &mut KeychainTxOutIndex<TestKeychain>, index: &(TestKeychain, u32)) {
    let op = OutPoint {
        txid: h!("dangling"),
        vout: 0,
    };
    let txout = TxOut {
        value: 1,
        script_pubkey: tracker
            .spk_at_index(index)
            .expect("bad keychain, or script not derived")
            .clone(),
    };
    tracker.scan_txout(op, &txout);
    assert!(tracker.is_used(index));
}

#[test]
fn test_store_all_up_to() {
    let mut txout_index = init_txout_index();
    let derive_to: BTreeMap<_, _> =
        [(TestKeychain::External, 12), (TestKeychain::Internal, 24)].into();
    assert_eq!(
        txout_index.store_all_up_to(&derive_to),
        [(TestKeychain::External, 12), (TestKeychain::Internal, 24)].into()
    );
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

    assert_eq!(
        txout_index.store_up_to(&TestKeychain::External, 3),
        [(TestKeychain::External, 3)].into(),
    );
    txout_index.scan_txout(
        OutPoint::default(),
        &TxOut {
            value: 420,
            script_pubkey: external_spk3,
        },
    );

    assert_eq!(
        txout_index.pad_all_with_unused(5),
        [(TestKeychain::External, 8), (TestKeychain::Internal, 4)].into(),
    );
    assert_eq!(
        txout_index.derivation_indices(),
        [(TestKeychain::External, 8), (TestKeychain::Internal, 4)].into()
    );
}

#[test]
fn test_wildcard_derivations() {
    let mut txout_index = init_txout_index();

    // Stage 1
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
    assert_eq!(
        txout_index.derive_new(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex(
                    "5120a60869f0dbcf1dc659c9cecbaf8050135ea9e8cdc487053f1dc6880949dc684c"
                )
                .unwrap()
            ),
            [(TestKeychain::External, 0)].into()
        )
    );
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex(
                    "5120a60869f0dbcf1dc659c9cecbaf8050135ea9e8cdc487053f1dc6880949dc684c"
                )
                .unwrap()
            ),
            [].into()
        )
    );

    // Stage - 2
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
        .for_each(|index| mark_used(&mut txout_index, &(TestKeychain::External, index)));

    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (26, true)
    );
    assert_eq!(
        txout_index.derive_new(&TestKeychain::External),
        (
            (
                26,
                &Script::from_hex(
                    "512017561301dafcfe66d9a757d9907b05646c9c1fcab57069f7917d0af3116661ee"
                )
                .unwrap()
            ),
            [(TestKeychain::External, 26)].into()
        )
    );
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External),
        (
            (
                16,
                &Script::from_hex(
                    "5120c0926db156acc59de0a5685b47ff6663edc37df9bbde9cf504ebe39568cb4644"
                )
                .unwrap()
            ),
            [].into()
        )
    );

    // Stage -3
    // - Use all the derived till 26.
    // - next_unused() = ((27, <spk>), DerivationAdditions)
    (0..=26)
        .into_iter()
        .for_each(|index| mark_used(&mut txout_index, &(TestKeychain::External, index)));

    assert_eq!(
        txout_index.next_unused(&TestKeychain::External),
        (
            (
                27,
                &Script::from_hex(
                    "512090f6ddcb6c1bd1482166b941a86e9663b44d889ab7e5ffce1c5185658d894bf3"
                )
                .unwrap()
            ),
            [(TestKeychain::External, 27)].into()
        )
    );

    /*
    Stage 4 and Stage 5 is commented out as they test behavior at numeric bound
    which takes a long time to run an can exhaust memory pretty quickly.

    included here for documenting only.

    // Stage  4
    // Derived scripts at numeric bound.
    //
    // - next_derivation_index() = (Max, false)
    // - derive_new() == ((Max, <spk>), DerivationAdditions::is_empty())
    // - next_unused() == ((27, <spk>), DerivationAdditions::is_empty())
    let _ = txout_index.store_up_to(&TestKeychain::External, u32::MAX);

    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (u32::MAX, false)
    );
    assert_eq!(
        txout_index.derive_new(&TestKeychain::External).1, [].into()
    );
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External).1, [].into()
    );

    // Stage 5
    // Used scripts at numeric bound. No unused script left. Can't derive either.
    // - next_unused() == ((Max, <spk>), DerivationAdditions::is_empty())
    (0..=u32::MAX).into_iter().for_each(|index| mark_used(&mut txout_index, &(TestKeychain::External, index)));
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External).1, [].into()
    );

    */
}

#[test]
fn test_non_wildcard_derivations() {
    let mut txout_index = KeychainTxOutIndex::<TestKeychain>::default();

    let secp = bitcoin::secp256k1::Secp256k1::signing_only();
    let (no_wildcard_descriptor, _) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "wpkh([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/0)").unwrap();

    txout_index.add_keychain(TestKeychain::External, no_wildcard_descriptor);

    // Stage 1
    // nothing derived
    // no unused scripts
    //
    // - next_derivation_index() = (0, true)
    // - derive_new() = ((0, <spk>), DerivationAdditions)
    // - next_unused() = ((0, <spk>), DerivationAdditions::is_empty())
    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (0, true)
    );
    assert_eq!(
        txout_index.derive_new(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex("0014dd494ba8f622f9ae256d2d58bc48214a6d4819f7").unwrap()
            ),
            [(TestKeychain::External, 0)].into()
        )
    );
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex("0014dd494ba8f622f9ae256d2d58bc48214a6d4819f7").unwrap()
            ),
            [].into()
        )
    );

    // Stage 2
    // Mark the single spk as used
    //
    // - next_derivation_index() == (0, false)
    // - derive_new() = ((0, <spk>), DerivationAdditions::is_empty())
    // - next_unused() = ((0, <spk>), DerivationAdditions::is_empty())

    mark_used(&mut txout_index, &(TestKeychain::External, 0));

    assert_eq!(
        txout_index.next_derivation_index(&TestKeychain::External),
        (0, false)
    );
    assert_eq!(
        txout_index.derive_new(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex("0014dd494ba8f622f9ae256d2d58bc48214a6d4819f7").unwrap()
            ),
            [].into()
        )
    );
    assert_eq!(
        txout_index.next_unused(&TestKeychain::External),
        (
            (
                0,
                &Script::from_hex("0014dd494ba8f622f9ae256d2d58bc48214a6d4819f7").unwrap()
            ),
            [].into()
        )
    );
}
