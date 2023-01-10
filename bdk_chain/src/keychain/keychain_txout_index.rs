use crate::{
    collections::*,
    miniscript::{Descriptor, DescriptorPublicKey},
    ForEachTxout, SpkTxOutIndex,
};
use bitcoin::{secp256k1::Secp256k1, OutPoint, Script, TxOut};
use core::{fmt::Debug, ops::Deref};

/// A convenient wrapper around [`SpkTxOutIndex`] that sets the script pubkeys basaed on a miniscript
/// [`Descriptor<DescriptorPublicKey>`][`Descriptor`]s.
///
/// ## Synopsis
///
/// ```
/// use bdk_chain::keychain::KeychainTxOutIndex;
/// # use bdk_chain::{ miniscript::{Descriptor, DescriptorPublicKey} };
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
/// # let secp = bdk_chain::bitcoin::secp256k1::Secp256k1::signing_only();
/// # let (external_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").unwrap();
/// # let (internal_descriptor,_) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").unwrap();
/// # let descriptor_for_user_42 = external_descriptor.clone();
///
/// txout_index.add_keychain(MyKeychain::External, external_descriptor);
/// txout_index.add_keychain(MyKeychain::Internal, internal_descriptor);
/// txout_index.add_keychain(MyKeychain::MyAppUser { user_id: 42 }, descriptor_for_user_42);
///
/// let new_spk_for_user =  txout_index.derive_new(&MyKeychain::MyAppUser { user_id: 42 });
/// ```
///
/// [`Ord`]: core::cmp::Ord
/// [`SpkTxOutIndex`]: crate::spk_txout_index::SpkTxOutIndex
/// [`Descriptor`]: crate::miniscript::Descriptor
#[derive(Clone, Debug)]
pub struct KeychainTxOutIndex<K> {
    inner: SpkTxOutIndex<(K, u32)>,
    keychains: BTreeMap<K, Descriptor<DescriptorPublicKey>>,
}

impl<K> Default for KeychainTxOutIndex<K> {
    fn default() -> Self {
        Self {
            inner: SpkTxOutIndex::default(),
            keychains: BTreeMap::default(),
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
    /// Scans an object containing many txouts.
    ///
    /// Typically this is used in two situations:
    ///
    /// 1. After loading transaction data from disk you may scan over all the txouts to restore all
    /// your txouts.
    /// 2. When getting new data from the chain you usually scan it before incoporating it into your chain state.
    ///
    /// See [`ForEachTxout`] for the types that support this.
    ///
    /// [`ForEachTxout`]: crate::ForEachTxout
    pub fn scan(&mut self, txouts: &impl ForEachTxout) {
        self.inner.scan(txouts);
    }

    /// Scan a single `TxOut` for a matching script pubkey.
    ///
    /// If it matches the index will store and index it.
    pub fn scan_txout(&mut self, op: OutPoint, txout: &TxOut) {
        self.inner.scan_txout(op, &txout);
    }

    pub fn inner(&self) -> &SpkTxOutIndex<(K, u32)> {
        &self.inner
    }

    /// Get the internal map of keychains to their descriptors.
    pub fn keychains(&self) -> &BTreeMap<K, Descriptor<DescriptorPublicKey>> {
        &self.keychains
    }

    /// Add a keychain to the tracker's `txout_index` with a descriptor to derive addresses for it.
    ///
    /// Adding a keychain means you will be able to derive new script pubkeys under that keychain
    /// and the txout index will discover transaction outputs with those script pubkeys.
    pub fn add_keychain(&mut self, keychain: K, descriptor: Descriptor<DescriptorPublicKey>) {
        // TODO: panic if already different descriptor at that keychain
        self.keychains.insert(keychain, descriptor);
    }

    /// Generates iterators for the script pubkeys of every keychain.
    ///
    /// Convienience method for calling [`script_pubkeys_of_keychain`] on each keychain.
    ///
    /// [`script_pubkeys_of_keychain`]: Self::script_pubkeys_of_keychain
    pub fn script_pubkeys_of_all_keychains(
        &self,
    ) -> BTreeMap<K, impl Iterator<Item = (u32, Script)> + Clone> {
        self.keychains()
            .iter()
            .map(|(keychain, descriptor)| {
                (
                    keychain.clone(),
                    descriptor_into_script_iter(descriptor.clone()),
                )
            })
            .collect()
    }

    /// Iterates over the script pubkeys derived and stored by this index under `keychain`
    pub fn script_pubkeys_of_keychain(
        &self,
        keychain: &K,
    ) -> impl DoubleEndedIterator<Item = (u32, &Script)> {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .map(|((_, derivation_index), spk)| (*derivation_index, spk))
    }

    /// Get the derivation index after the current one
    pub fn next_derivation_index(&self, keychain: &K) -> u32 {
        self.derivation_index(keychain)
            .map(|index| index + 1)
            .unwrap_or(0)
    }

    /// Get the current derivation index. This is the highest index in the keychain we have stored.
    pub fn derivation_index(&self, keychain: &K) -> Option<u32> {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .last()
            .map(|((_, index), _)| *index)
    }

    /// Gets the current derivation index for each keychain in the index.
    pub fn derivation_indices(&self) -> BTreeMap<K, u32> {
        self.keychains()
            .keys()
            .filter_map(|keychain| Some((keychain.clone(), self.derivation_index(&keychain)?)))
            .collect()
    }

    /// Convenience method to call [`derive_spks_up_to`] on several keychains.
    ///
    /// Returns whether any new script pubkeys were derived (or if they had already all been
    /// stored).
    ///
    /// [`derive_spks_up_to`]: Self::store_up_to
    pub fn store_all_up_to(&mut self, keychains: &BTreeMap<K, u32>) -> bool {
        let mut changed = false;
        for (keychain, &index) in keychains {
            changed |= self.store_up_to(keychain, index);
        }
        changed
    }

    /// Derives script pubkeys from the descriptor **up to and including** `up_to` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new script pubkeys were derived. This will be false when they had already all been
    /// stored or wheen the `keychain` itself was never added to the index.
    pub fn store_up_to(&mut self, keychain: &K, up_to: u32) -> bool {
        let descriptor = match self.keychains.get(&keychain) {
            Some(descriptor) => descriptor,
            None => return false,
        };

        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => up_to,
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
    ///
    /// ## Panics
    ///
    /// Panics if the `keychain` has not been added to the index.
    pub fn derive_new(&mut self, keychain: &K) -> (u32, &Script) {
        let secp = Secp256k1::verification_only();
        let next_derivation_index = self.next_derivation_index(keychain);
        let descriptor = self
            .keychains
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

    /// Gets the next usued script pubkey in the keychain i.e. the script pubkey with the lowest index that has not been used yet.
    ///
    /// ## Panics
    ///
    /// Panics if `keychain` has never been added to the index
    pub fn next_unused(&mut self, keychain: &K) -> (u32, &Script) {
        let need_new = self.keychain_unused(keychain).next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(keychain)
        } else {
            self.keychain_unused(keychain).next().unwrap()
        }
    }

    /// Convenience method to call [`pad_with_unused`] on all keychains.
    ///
    /// Returns whether any new scripts were derived.
    ///
    /// [`pad_with_unused`]: Self::pad_with_unused
    pub fn pad_all_with_unused(&mut self, pad_len: u32) -> bool {
        let mut changed = false;
        let keychains = self
            .keychains
            .keys()
            .cloned()
            .collect::<alloc::vec::Vec<_>>();
        for keychain in keychains {
            changed |= self.pad_with_unused(&keychain, pad_len);
        }
        changed
    }

    /// Derives and stores `pad_len` new script pubkeys after the last active index of `keychain`.
    ///
    /// This is useful when scanning blockchain data for transaction outputs belonging to the
    /// keychain since the derivation index of new transactions is likely to be higher than the
    /// current last active index.
    pub fn pad_with_unused(&mut self, keychain: &K, pad_len: u32) -> bool {
        let up_to = self
            .last_active_index(keychain)
            .map(|i| i.saturating_add(pad_len))
            .or(pad_len.checked_sub(1));
        match up_to {
            Some(up_to) => self.store_up_to(keychain, up_to),
            None => false,
        }
    }

    /// Iterates over all unused script pubkeys for a `keychain` that have been stored in the index.
    pub fn keychain_unused(&self, keychain: &K) -> impl DoubleEndedIterator<Item = (u32, &Script)> {
        let range = (keychain.clone(), u32::MIN)..(keychain.clone(), u32::MAX);
        self.inner
            .unused(range)
            .map(|((_, i), script)| (*i, script))
    }

    /// Iterates over all the [`OutPoint`] that have a `TxOut` with a script pubkey derived from `keychain`
    pub fn keychain_txouts(
        &self,
        keychain: &K,
    ) -> impl DoubleEndedIterator<Item = (u32, OutPoint)> + '_ {
        self.inner
            .outputs_in_range((keychain.clone(), u32::MIN)..(keychain.clone(), u32::MAX))
            .map(|((_, i), op)| (*i, op))
    }

    /// The highest derivation index of `keychain` that the index has found a `TxOut` with its script pubkey.
    pub fn last_active_index(&self, keychain: &K) -> Option<u32> {
        self.keychain_txouts(keychain).last().map(|(i, _)| i)
    }

    /// The highest derivation index of each keychain that the index has found a `TxOut` with its script pubkey.
    pub fn last_active_indicies(&self) -> BTreeMap<K, u32> {
        self.keychains
            .iter()
            .filter_map(|(keychain, _)| {
                self.last_active_index(keychain)
                    .map(|index| (keychain.clone(), index))
            })
            .collect()
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
