use crate::{
    collections::*,
    miniscript::{Descriptor, DescriptorPublicKey},
    ForEachTxout, SpkTxOutIndex,
};
use bitcoin::{secp256k1::Secp256k1, OutPoint, Script, TxOut};
use core::{fmt::Debug, ops::Deref};

use super::DerivationAdditions;

const DERIVED_KEY_COUNT: u32 = 1 << 31;

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
    /// 2. When getting new data from the chain you usually scan it before incorporating it into
    /// your chain state (i.e. `SparseChain`, `ChainGraph`).
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

    /// Generates script pubkey iterators for every `keychain`. The iterators iterate over all
    /// derivable scripts.
    pub fn scripts_of_all_keychains(
        &self,
    ) -> BTreeMap<K, impl Iterator<Item = (u32, Script)> + Clone> {
        self.keychains
            .iter()
            .map(|(keychain, descriptor)| {
                (
                    keychain.clone(),
                    descriptor_into_script_iter(descriptor.clone()),
                )
            })
            .collect()
    }

    /// Generates a script pubkey iterator for the given `keychain`'s descriptor (if exists). The
    /// iterator iterates over all derivable scripts of the keychain's descriptor.
    pub fn scripts_of_keychain(
        &self,
        keychain: &K,
    ) -> Option<impl Iterator<Item = (u32, Script)> + Clone> {
        self.keychains
            .get(keychain)
            .cloned()
            .map(descriptor_into_script_iter)
    }

    /// Iterates over the script pubkeys derived and stored by this index of all keychains.
    pub fn stored_scripts_of_all_keychains(
        &self,
    ) -> BTreeMap<K, impl Iterator<Item = (u32, &Script)> + Clone> {
        self.keychains
            .keys()
            .map(|keychain| (keychain.clone(), self.stored_scripts_of_keychain(keychain)))
            .collect()
    }

    /// Iterates over the script pubkeys derived and stored by this index under `keychain`.
    pub fn stored_scripts_of_keychain(
        &self,
        keychain: &K,
    ) -> impl DoubleEndedIterator<Item = (u32, &Script)> + Clone {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .map(|((_, derivation_index), spk)| (*derivation_index, spk))
    }

    /// Get the next derivation index for `keychain`.
    ///
    /// The second field in the returned tuple represents whether the next derivation index is new.
    /// There are two scenarios where the next derivation index is reused (not new):
    ///
    /// 1. The keychain's descriptor has no wildcard, and a script has already been derived.
    /// 2. The number of derived scripts has already reached 2^31 (refer to BIP-32).
    ///
    /// Not checking the second field of the tuple may result in address reuse.
    ///
    /// ## Panics
    ///
    /// Panics if the `keychain` does not exist.
    pub fn next_derivation_index(&self, keychain: &K) -> (u32, bool) {
        // we can only get the next index if wildcard exists
        let has_wildcard = self
            .keychains
            .get(keychain)
            .expect(&format!("keychain {:?} does not exist", keychain))
            .has_wildcard();

        match self.derivation_index(keychain) {
            // if there is no index, next_index is always 0
            None => (0, true),
            // descriptors without wildcards can only have one index
            Some(_) if !has_wildcard => (0, false),
            // derivation index must be < 2^31 (BIP-32)
            Some(index) if index >= DERIVED_KEY_COUNT => unreachable!("index is out of bounds"),
            Some(index) if index == DERIVED_KEY_COUNT - 1 => (index, false),
            // get next derivation index
            Some(index) => (index + 1, true),
        }
    }

    /// Get the current derivation index for `keychain`.
    ///
    /// This is the highest index we have stored for `keychain`.
    pub fn derivation_index(&self, keychain: &K) -> Option<u32> {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .last()
            .map(|((_, index), _)| *index)
    }

    /// Get the current derivation index for each keychain.
    ///
    /// Keychains with no indicies derived will not be included in the returned [`BTreeMap`].
    pub fn derivation_indices(&self) -> BTreeMap<K, u32> {
        self.keychains()
            .keys()
            .filter_map(|keychain| Some((keychain.clone(), self.derivation_index(&keychain)?)))
            .collect()
    }

    /// Convenience method to call [`Self::store_up_to`] on several keychains.
    pub fn store_all_up_to(&mut self, keychains: &BTreeMap<K, u32>) -> DerivationAdditions<K> {
        let mut additions = DerivationAdditions::default();
        for (keychain, &index) in keychains {
            additions.append(self.store_up_to(keychain, index));
        }
        additions
    }

    /// Derives script pubkeys from the descriptor **up to and including** `up_to` and stores them
    /// (if necessary).
    ///
    /// Returns [`DerivationAdditions`] for any new `script_pubkey`s that has been added. If no
    /// script pubkeys are added, or if `keychain` does not exist, [`DerivationAdditions`] will be
    /// empty.
    pub fn store_up_to(&mut self, keychain: &K, up_to: u32) -> DerivationAdditions<K> {
        let descriptor = match self.keychains.get(&keychain) {
            Some(descriptor) => descriptor,
            None => return Default::default(),
        };

        let end = match descriptor.has_wildcard() {
            false => 0,
            true => up_to,
        };
        let (next_to_derive, can_derive) = self.next_derivation_index(keychain);
        if !can_derive || next_to_derive > end {
            return Default::default();
        }

        let secp = Secp256k1::verification_only();
        for index in next_to_derive..=end {
            let spk = match descriptor
                .at_derivation_index(index)
                .derived_descriptor(&secp)
            {
                Ok(derived_desciptor) => derived_desciptor.script_pubkey(),
                Err(_) => continue,
            };
            self.inner
                .insert_script_pubkey((keychain.clone(), index), spk);
        }

        let mut additions = BTreeMap::default();
        additions.insert(keychain.clone(), end);
        DerivationAdditions(additions)
    }

    /// Derives a new script pubkey for `keychain`.
    ///
    /// Returns the derivation index of the derived script pubkey, the derived script pubkey and a
    /// [`DerivationAdditions`] which represents changes in the internal index (if any).
    ///
    /// When a new script cannot be derived, we return the last derived script and an empty
    /// [`DerivationAdditions`]. There are two scenarios when a new script pubkey cannot be derived:
    ///
    ///  1. The descriptor has no wildcard and already has one script derived.
    ///  2. The descriptor has already derived scripts up to the numeric bound.
    ///
    /// ## Panics
    ///
    /// Panics if the `keychain` does not exist.
    pub fn derive_new(&mut self, keychain: &K) -> ((u32, &Script), DerivationAdditions<K>) {
        let (next_index, _) = self.next_derivation_index(keychain);
        let additions = self.store_up_to(keychain, next_index);
        let script = self
            .inner
            .spk_at_index(&(keychain.clone(), next_index))
            .expect("script must already be stored");
        ((next_index, script), additions)
    }

    /// Gets the next unused script pubkey in the keychain. I.e. the script pubkey with the lowest
    /// index that has not been used yet.
    ///
    /// This will derive and store a new script pubkey if no more unused script pubkeys exist.
    ///
    /// If the descriptor has no wildcard and already has a used script pubkey, or if a descriptor
    /// has used all scripts up to the derivation bounds, the last derived script pubkey will be
    /// returned.
    ///
    /// ## Panics
    ///
    /// Panics if `keychain` has never been added to the index
    pub fn next_unused(&mut self, keychain: &K) -> ((u32, &Script), DerivationAdditions<K>) {
        let need_new = self.keychain_unused(keychain).next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(keychain)
        } else {
            (
                self.keychain_unused(keychain)
                    .next()
                    .expect("we already know next exists"),
                DerivationAdditions::default(),
            )
        }
    }

    /// Marks the script pubkey at `index` as used even though it hasn't seen an output with it.
    /// This only has an effect when the `index` had been added to `self` already and was unused.
    ///
    /// This is useful when you want to reserve a script pubkey for something but don't want to add
    /// the transaction output using it to the index yet. Other callers will consider `index` on
    /// `keychain` used until you call [`unmark_used`].
    ///
    /// [`unmark_used`]: Self::unmark_used
    pub fn mark_used(&mut self, keychain: &K, index: u32) {
        self.inner.mark_used(&(keychain.clone(), index))
    }

    /// Undoes the effect of [`mark_used`].
    ///
    /// Note that if `self` has scanned an output with this script pubkey then this will have no
    /// effect.
    ///
    /// [`mark_used`]: Self::mark_used
    pub fn unmark_used(&mut self, keychain: &K, index: u32) {
        self.inner.unmark_used(&(keychain.clone(), index));
    }

    /// Convenience method to call [`Self::pad_keychain_with_unused`] on all keychains.
    ///
    /// Returns whether any new scripts were derived.
    pub fn pad_all_with_unused(&mut self, pad_len: u32) -> DerivationAdditions<K> {
        let keychains = self.keychains.clone().into_keys();

        let mut additions = DerivationAdditions::default();
        for keychain in keychains {
            additions.append(self.pad_keychain_with_unused(&keychain, pad_len));
        }
        additions
    }

    /// Derives and stores `pad_len` new script pubkeys after the last active index of `keychain`.
    ///
    /// This is useful when scanning blockchain data for transaction outputs belonging to the
    /// keychain since the derivation index of new transactions is likely to be higher than the
    /// current last active index.
    pub fn pad_keychain_with_unused(
        &mut self,
        keychain: &K,
        pad_len: u32,
    ) -> DerivationAdditions<K> {
        let up_to = self
            .last_active_index(keychain)
            .map(|i| i.saturating_add(pad_len))
            .or(pad_len.checked_sub(1));
        match up_to {
            Some(up_to) => self.store_up_to(keychain, up_to),
            None => Default::default(),
        }
    }

    /// Iterates over all unused script pubkeys for a `keychain` that have been stored in the index.
    pub fn keychain_unused(&self, keychain: &K) -> impl DoubleEndedIterator<Item = (u32, &Script)> {
        let range = (keychain.clone(), u32::MIN)..(keychain.clone(), u32::MAX);
        self.inner
            .unused(range)
            .map(|((_, i), script)| (*i, script))
    }

    /// Iterates over all the [`OutPoint`] that have a `TxOut` with a script pubkey derived from
    /// `keychain`.
    pub fn keychain_txouts(
        &self,
        keychain: &K,
    ) -> impl DoubleEndedIterator<Item = (u32, OutPoint)> + '_ {
        self.inner
            .outputs_in_range((keychain.clone(), u32::MIN)..(keychain.clone(), u32::MAX))
            .map(|((_, i), op)| (*i, op))
    }

    /// Returns the highest derivation index of the `keychain` where [`KeychainTxOutIndex`] has
    /// found a [`TxOut`] with it's script pubkey.
    pub fn last_active_index(&self, keychain: &K) -> Option<u32> {
        self.keychain_txouts(keychain).last().map(|(i, _)| i)
    }

    /// Returns the highest derivation index of each keychain that [`KeychainTxOutIndex`] has found
    /// a [`TxOut`] with it's script pubkey.
    pub fn last_active_indicies(&self) -> BTreeMap<K, u32> {
        self.keychains
            .iter()
            .filter_map(|(keychain, _)| {
                self.last_active_index(keychain)
                    .map(|index| (keychain.clone(), index))
            })
            .collect()
    }

    /// Applies the derivation additions to the [`KeychainTxOutIndex`], extending the number of
    /// derived scripts per keychain, as specified in the `additions`.
    pub fn apply_additions(&mut self, additions: DerivationAdditions<K>) {
        let _ = self.store_all_up_to(&additions.0);
    }
}

fn descriptor_into_script_iter(
    descriptor: Descriptor<DescriptorPublicKey>,
) -> impl Iterator<Item = (u32, Script)> + Clone + Send {
    let secp = Secp256k1::verification_only();
    let end = if descriptor.has_wildcard() {
        // we only iterate over non-hardened indexes
        DERIVED_KEY_COUNT - 1
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
