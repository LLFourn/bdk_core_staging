use bdk_core::{
    bitcoin::{secp256k1::Secp256k1, OutPoint, Script, TxOut},
    collections::*,
    ForEachTxout, SpkTxOutIndex,
};
use core::{fmt::Debug, ops::Deref};
use miniscript::{Descriptor, DescriptorPublicKey};

/// A convenient wrapper around [`SpkTxOutIndex`] that sets the script pubkeys basaed on a miniscript
/// [`Descriptor<DescriptorPublicKey>`][`Descriptor`]s.
///
/// ## Synopsis
///
/// ```
/// use bdk_keychain::KeychainTxOutIndex;
/// # use bdk_keychain::{ miniscript::{Descriptor, DescriptorPublicKey} };
/// # use core::str::FromStr;
///
/// // imagine our service has internal and external addresses but also addresses for users
/// #[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
/// enum MyKeychain {
///     External,
///     Internal,
///     MyAppUser {
///         user_id: u32
///     }
/// }
///
/// let mut txout_index = KeychainTxOutIndex::<MyKeychain>::default();
///
/// # let secp = bdk_keychain::bdk_core::bitcoin::secp256k1::Secp256k1::signing_only();
/// # let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
/// # let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();
/// # let descriptor_for_user_42 = external_descriptor.clone();
///
/// txout_index.add_keychain(MyKeychain::External, external_descriptor);
/// txout_index.add_keychain(MyKeychain::Internal, internal_descriptor);
/// txout_index.add_keychain(MyKeychain::MyAppUser { user_id: 42 }, descriptor_for_user_42);
///
/// let new_spk_for_user =  txout_index.derive_new(MyKeychain::MyAppUser { user_id: 42 });
/// ```
///
/// [`Ord`]: core::cmp::Ord
/// [`SpkTxOutIndex`]: crate::SpkTxOutIndex
/// [`Descriptor`]: miniscript::Descriptor
#[derive(Clone, Debug)]
pub struct KeychainTxOutIndex<K> {
    inner: SpkTxOutIndex<(K, u32)>,
    descriptors: BTreeMap<K, Descriptor<DescriptorPublicKey>>,
}

impl<K> Default for KeychainTxOutIndex<K> {
    fn default() -> Self {
        Self {
            inner: SpkTxOutIndex::default(),
            descriptors: BTreeMap::default(),
        }
    }
}

impl<K> Deref for KeychainTxOutIndex<K> {
    type Target = SpkTxOutIndex<(K, u32)>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K: Clone + Ord + Debug> KeychainTxOutIndex<K> {
    pub fn scan(&mut self, txouts: &impl ForEachTxout) {
        self.inner.scan(txouts);
    }

    pub fn scan_txout(&mut self, op: OutPoint, txout: &TxOut) {
        self.inner.scan_txout(op, &txout)
    }

    pub fn inner(&self) -> &SpkTxOutIndex<(K, u32)> {
        &self.inner
    }

    pub fn keychains(
        &self,
        range: impl core::ops::RangeBounds<K>,
    ) -> impl DoubleEndedIterator<Item = (K, &Descriptor<DescriptorPublicKey>)> {
        self.descriptors
            .range(range)
            .map(|(keychain, descriptor)| (keychain.clone(), descriptor))
    }

    pub fn add_keychain(&mut self, keychain: K, descriptor: Descriptor<DescriptorPublicKey>) {
        // TODO: panic if already different descriptor at that keychain
        self.descriptors.insert(keychain, descriptor);
    }

    /// Generates a map of keychain to an iterator over all that keychain's
    pub fn start_wallet_scan(&self) -> BTreeMap<K, impl Iterator<Item = (u32, Script)> + Clone> {
        self.keychains(..)
            .map(|(keychain, descriptor)| {
                (keychain, descriptor_into_script_iter(descriptor.clone()))
            })
            .collect()
    }

    pub fn descriptor(&self, keychain: &K) -> &Descriptor<DescriptorPublicKey> {
        self.descriptors
            .get(&keychain)
            .expect("keychain does not exist")
    }

    /// Get the next
    pub fn next_derivation_index(&self, keychain: &K) -> u32 {
        self.derivation_index(keychain)
            .map(|index| index + 1)
            .unwrap_or(0)
    }

    pub fn derivation_index(&self, keychain: &K) -> Option<u32> {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .map(|((_, index), _)| *index)
            .last()
    }

    pub fn derivation_indices(&self) -> BTreeMap<K, u32> {
        self.keychains(..)
            .filter_map(|(keychain, _)| Some((keychain.clone(), self.derivation_index(&keychain)?)))
            .collect()
    }

    pub fn derive_all_spks(&mut self, keychains: &BTreeMap<K, u32>) -> bool {
        keychains
            .into_iter()
            .any(|(keychain, index)| self.derive_spks(keychain, *index))
    }

    /// Derives script pubkeys from the descriptor **up to and including** `end` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn derive_spks(&mut self, keychain: &K, index: u32) -> bool {
        let descriptor = self
            .descriptors
            .get(&keychain)
            .expect(&format!("no descriptor for keychain {:?}", keychain));
        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => index,
        };
        let next_to_derive = self.next_derivation_index(keychain);
        if next_to_derive > end {
            return false;
        }

        for index in next_to_derive..=end {
            let spk = descriptor
                .at_derivation_index(index)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey();
            self.inner.add_spk((keychain.clone(), index), spk);
        }

        true
    }

    /// Derives a new script pubkey for a keychain.
    ///
    /// The index returns a new script pubkey for each call to this method and stores it internally
    /// so it will be able to find transactions related to it.
    pub fn derive_new(&mut self, keychain: &K) -> (u32, &Script) {
        let secp = Secp256k1::verification_only();
        let next_derivation_index = self.next_derivation_index(keychain);
        let descriptor = self
            .descriptors
            .get(&keychain)
            .expect(&format!("no descriptor for keychain {:?}", keychain));

        let new_spk = descriptor
            .at_derivation_index(next_derivation_index as u32)
            .derived_descriptor(&secp)
            .expect("the descriptor cannot need hardened derivation")
            .script_pubkey();

        let index = (keychain.clone(), next_derivation_index);
        self.inner.add_spk(index.clone(), new_spk);
        let new_spk = self
            .inner
            .script_pubkeys()
            .get(&index)
            .expect("we just added it");
        (next_derivation_index, new_spk)
    }

    pub fn derive_next_unused(&mut self, keychain: &K) -> (u32, &Script) {
        let need_new = self
            .inner
            .iter_unused()
            .filter(|((kc, _), _)| kc == keychain)
            .next()
            .is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(keychain)
        } else {
            self.inner
                .iter_unused()
                .filter(|((kc, _), _)| kc == keychain)
                .map(|((_, i), script)| (i, script))
                .next()
                .unwrap()
        }
    }
}

fn descriptor_into_script_iter(
    descriptor: Descriptor<DescriptorPublicKey>,
) -> impl Iterator<Item = (u32, Script)> + Clone + Send {
    let secp = Secp256k1::verification_only();
    let end = if descriptor.has_wildcard() {
        // Because we only iterate over non-hardened indexes there are 2^31 values
        (1 << 31) - 1
    } else {
        0
    };

    (0..=end).map(move |i| {
        (
            i,
            descriptor
                .at_derivation_index(i)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey(),
        )
    })
}
