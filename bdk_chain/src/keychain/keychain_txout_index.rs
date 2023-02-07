use crate::{
    collections::*,
    miniscript::{Descriptor, DescriptorPublicKey},
    ForEachTxout, SpkTxOutIndex,
};
use alloc::vec::Vec;
use bitcoin::{
    secp256k1::{self, Secp256k1},
    OutPoint, Script, TxOut,
};
use core::{fmt::Debug, ops::Deref};

use super::DerivationAdditions;

const DERIVED_KEY_COUNT: u32 = 1 << 31;

/// A convenient wrapper around [`SpkTxOutIndex`] that relates script pubkeys to miniscript public
/// [`Descriptor`]s.
///
/// Descriptors are referenced by the provided keychain generic (`K`).
///
/// Script pubkeys for a descriptor are stored chronologically from index 0. I.e. If the last stored
/// index of a descriptor is 5, scripts of indices 0 to 4 are also guaranteed to be stored.
///
/// Methods that may result in changes to the number of stored script pubkeys will return
/// [`DerivationAdditions`] to reflect the changes. This can be persisted for future recovery.
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
/// txout_index.add_keychain(MyKeychain::External, external_descriptor);
/// txout_index.add_keychain(MyKeychain::Internal, internal_descriptor);
/// txout_index.add_keychain(MyKeychain::MyAppUser { user_id: 42 }, descriptor_for_user_42);
///
/// let new_spk_for_user = txout_index.new_script(&MyKeychain::MyAppUser{ user_id: 42 });
/// ```
///
/// [`Ord`]: core::cmp::Ord
/// [`SpkTxOutIndex`]: crate::spk_txout_index::SpkTxOutIndex
/// [`Descriptor`]: crate::miniscript::Descriptor
#[derive(Clone, Debug)]
pub struct KeychainTxOutIndex<K> {
    inner: SpkTxOutIndex<(K, u32)>,
    // descriptors of each keychain
    keychains: BTreeMap<K, Descriptor<DescriptorPublicKey>>,
    // last stored indexes
    last_stored: BTreeMap<K, u32>,
    // lookahead settings for each keychain
    lookahead: BTreeMap<K, u32>,
}

impl<K> Default for KeychainTxOutIndex<K> {
    fn default() -> Self {
        Self {
            inner: SpkTxOutIndex::default(),
            keychains: BTreeMap::default(),
            last_stored: BTreeMap::default(),
            lookahead: BTreeMap::default(),
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
    /// Scans an object for relevant outpoints, which are stored and indexed internally.
    ///
    /// If the matched script pubkey is part of the lookahead, the last stored index is updated for
    /// the script pubkey's keychain and the [`DerivationAdditions`] returned will reflect the
    /// change.
    ///
    /// Typically this method is used in two situations:
    ///
    /// 1. After loading transaction data from disk you may scan over all the txouts to restore all
    /// your txouts.
    /// 2. When getting new data from the chain you usually scan it before incorporating it into
    /// your chain state (i.e. `SparseChain`, `ChainGraph`).
    ///
    /// See [`ForEachTxout`] for the types that support this.
    ///
    /// [`ForEachTxout`]: crate::ForEachTxout
    pub fn scan(&mut self, txouts: &impl ForEachTxout) -> DerivationAdditions<K> {
        let mut additions = DerivationAdditions::<K>::default();
        txouts.for_each_txout(&mut |(op, txout)| additions.append(self.scan_txout(op, txout)));
        additions
    }

    /// Scan a single outpoint for a matching script pubkey.
    ///
    /// If it matches the index will store and index it.
    pub fn scan_txout(&mut self, op: OutPoint, txout: &TxOut) -> DerivationAdditions<K> {
        let mut additions = DerivationAdditions::default();
        if let Some((keychain, index)) = self.inner.scan_txout(op, txout).cloned() {
            additions.append(self.store_up_to(&keychain, index));
        }
        additions
    }

    /// Return a reference to the internal [`SpkTxOutIndex`].
    pub fn inner(&self) -> &SpkTxOutIndex<(K, u32)> {
        &self.inner
    }

    /// Return a reference to the internal map of keychain to descriptors.
    pub fn keychains(&self) -> &BTreeMap<K, Descriptor<DescriptorPublicKey>> {
        &self.keychains
    }

    /// Add a keychain to the tracker's `txout_index` with a descriptor to derive addresses for it.
    ///
    /// Adding a keychain means you will be able to derive new script pubkeys under that keychain
    /// and the txout index will discover transaction outputs with those script pubkeys.
    ///
    /// # Panics
    ///
    /// This will panic if a different `descriptor` is introduced to the same `keychain`.
    pub fn add_keychain(&mut self, keychain: K, descriptor: Descriptor<DescriptorPublicKey>) {
        let old_descriptor = &*self.keychains.entry(keychain).or_insert(descriptor.clone());
        assert_eq!(
            &descriptor, old_descriptor,
            "keychain already contains a different keychain"
        );
    }

    /// Return the lookahead setting for each keychain.
    ///
    /// Refer to [`set_lookahead`] for a deeper explanation on `lookahead`.
    ///
    /// [`set_lookahead`]: Self::set_lookahead
    pub fn lookaheads(&self) -> &BTreeMap<K, u32> {
        &self.lookahead
    }

    /// Convenience method to call [`set_lookahead`] for all keychains.
    ///
    /// [`set_lookahead`]: Self::set_lookahead
    pub fn set_all_lookaheads(&mut self, lookahead: u32) {
        let secp = Secp256k1::verification_only();
        for keychain in &self.keychains.keys().cloned().collect::<Vec<_>>() {
            self.lookahead.insert(keychain.clone(), lookahead);
            self.replenish_lookahead(&secp, &keychain);
        }
    }

    /// Set the lookahead count for `keychain`.
    ///
    /// The lookahead is the number of scripts to cache ahead of the last stored script index. This
    /// is useful during a scan via [`scan`] or [`scan_txout`].
    ///
    /// # Panics
    ///
    /// This will panic if `keychain` does not exist.
    ///
    /// [`scan`]: Self::scan
    /// [`scan_txout`]: Self::scan_txout
    pub fn set_lookahead(&mut self, keychain: &K, lookahead: u32) {
        self.lookahead.insert(keychain.clone(), lookahead);
        self.replenish_lookahead(&Secp256k1::verification_only(), keychain);
    }

    fn replenish_lookahead<C>(&mut self, secp: &Secp256k1<C>, keychain: &K)
    where
        C: secp256k1::Verification,
    {
        let descriptor = self.keychains.get(keychain).expect("keychain must exist");
        let next_index = self.last_stored.get(keychain).map_or(0, |v| *v + 1);
        let lookahead = self.lookahead.get(keychain).map_or(0, |v| *v);

        for index in next_index..next_index + lookahead {
            let spk = descriptor
                .at_derivation_index(index)
                .derived_descriptor(secp)
                .map(|desc| desc.script_pubkey());

            match spk {
                Ok(spk) => self
                    .inner
                    .insert_script_pubkey((keychain.clone(), index), spk),
                Err(_) => break,
            };
        }
    }

    /// Generates script pubkey iterators for every `keychain`. The iterators iterate over all
    /// derivable scripts.
    pub fn all_keychain_scripts(&self) -> BTreeMap<K, impl Iterator<Item = (u32, Script)> + Clone> {
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
    ///
    /// # Panics
    ///
    /// This will panic if `keychain` does not exist.
    pub fn keychain_scripts(&self, keychain: &K) -> impl Iterator<Item = (u32, Script)> + Clone {
        let descriptor = self
            .keychains
            .get(keychain)
            .expect("keychain must exist")
            .clone();
        descriptor_into_script_iter(descriptor)
    }

    /// Convenience method to get [`stored_scripts`] for all keychains.
    ///
    /// [`stored_scripts`]: Self::stored_scripts
    pub fn all_stored_scripts(&self) -> BTreeMap<K, impl Iterator<Item = (u32, &Script)> + Clone> {
        self.keychains
            .keys()
            .map(|keychain| (keychain.clone(), self.stored_scripts(keychain)))
            .collect()
    }

    /// Iterates over the script pubkeys derived and stored by this index under `keychain`.
    pub fn stored_scripts(
        &self,
        keychain: &K,
    ) -> impl DoubleEndedIterator<Item = (u32, &Script)> + Clone {
        let next_index = self.last_stored.get(keychain).map_or(0, |v| *v + 1);
        self.inner
            .script_pubkeys()
            .range((keychain.clone(), u32::MIN)..(keychain.clone(), next_index))
            .map(|((_, derivation_index), spk)| (*derivation_index, spk))
    }

    /// Get the next derivation index for `keychain`. This is the index after the last stored index.
    ///
    /// The second field in the returned tuple represents whether the next derivation index is new.
    /// There are two scenarios where the next derivation index is reused (not new):
    ///
    /// 1. The keychain's descriptor has no wildcard, and a script has already been derived.
    /// 2. The number of derived scripts has already reached 2^31 (refer to BIP-32).
    ///
    /// Not checking the second field of the tuple may result in address reuse.
    ///
    /// # Panics
    ///
    /// Panics if the `keychain` does not exist.
    pub fn next_index(&self, keychain: &K) -> (u32, bool) {
        let descriptor = self.keychains.get(keychain).expect("keychain must exist");
        let last_index = self.last_stored.get(keychain).cloned();

        // we can only get the next index if wildcard exists
        let has_wildcard = descriptor.has_wildcard();

        match last_index {
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

    /// Get the last derivation index for each keychain.
    ///
    /// Keychains with no stored indices will not be included in the returned [`BTreeMap`].
    pub fn last_stored_indices(&self) -> &BTreeMap<K, u32> {
        &self.last_stored
    }

    /// Get the last derivation index for `keychain`.
    ///
    /// This is the highest index we have stored for `keychain`.
    pub fn last_stored_index(&self, keychain: &K) -> Option<u32> {
        self.last_stored.get(keychain).cloned()
    }

    /// Convenience method to call [`Self::store_up_to`] on several keychains.
    pub fn store_all_up_to(&mut self, keychains: &BTreeMap<K, u32>) -> DerivationAdditions<K> {
        let mut additions = DerivationAdditions::default();
        for (keychain, &index) in keychains {
            additions.append(self.store_up_to(keychain, index));
        }
        additions
    }

    /// Stores script pubkeys from the descriptor **up to and including** `index`, unless the script
    /// pubkey is already stored.
    ///
    /// Returns [`DerivationAdditions`] for new script pubkeys that have been stored. If no
    /// script pubkeys are stored, [`DerivationAdditions`] will be empty.
    ///
    /// # Panics
    ///
    /// Panics if `keychain` does not exist.
    pub fn store_up_to(&mut self, keychain: &K, index: u32) -> DerivationAdditions<K> {
        let descriptor = self.keychains.get(keychain).expect("keychain must exist");

        let next_index = self.last_stored.get(keychain).map_or(0, |v| *v + 1);
        if index < next_index {
            return DerivationAdditions::default();
        }

        let lookahead = self.lookahead.get(keychain).map_or(0, |v| *v);

        let secp = Secp256k1::verification_only();
        let mut last_derived = None;

        if index < next_index + lookahead {
            last_derived = Some(index);
        } else {
            for new_index in next_index + lookahead..=index {
                let spk = match descriptor
                    .at_derivation_index(new_index)
                    .derived_descriptor(&secp)
                {
                    Ok(derived_desciptor) => derived_desciptor.script_pubkey(),
                    Err(_) => break,
                };
                if self
                    .inner
                    .insert_script_pubkey((keychain.clone(), new_index), spk)
                {
                    last_derived = Some(new_index);
                }
            }
        }

        match last_derived {
            Some(index) => {
                self.last_stored.insert(keychain.clone(), index);
                self.replenish_lookahead(&secp, keychain);
                DerivationAdditions([(keychain.clone(), index)].into())
            }
            None => DerivationAdditions::default(),
        }
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
    /// # Panics
    ///
    /// Panics if the `keychain` does not exist.
    pub fn new_script(&mut self, keychain: &K) -> ((u32, &Script), DerivationAdditions<K>) {
        let (next_index, _) = self.next_index(keychain);
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
    /// # Panics
    ///
    /// Panics if `keychain` has never been added to the index
    pub fn next_unused_script(&mut self, keychain: &K) -> ((u32, &Script), DerivationAdditions<K>) {
        let need_new = self.unused_scripts(keychain).next().is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.new_script(keychain)
        } else {
            (
                self.unused_scripts(keychain)
                    .next()
                    .expect("we already know next exists"),
                DerivationAdditions::default(),
            )
        }
    }

    /// Marks the script pubkey at `index` as used even though it hasn't seen an output with it.
    /// This only has an effect when the `index` had been added to `self` already and was unused.
    ///
    /// Returns whether the `index` was originally present as `unused`.
    ///
    /// This is useful when you want to reserve a script pubkey for something but don't want to add
    /// the transaction output using it to the index yet. Other callers will consider `index` on
    /// `keychain` used until you call [`unmark_used`].
    ///
    /// [`unmark_used`]: Self::unmark_used
    pub fn mark_used(&mut self, keychain: &K, index: u32) -> bool {
        self.inner.mark_used(&(keychain.clone(), index))
    }

    /// Undoes the effect of [`mark_used`]. Returns whether the `index` is inserted back into
    /// `unused`.
    ///
    /// Note that if `self` has scanned an output with this script pubkey then this will have no
    /// effect.
    ///
    /// [`mark_used`]: Self::mark_used
    pub fn unmark_used(&mut self, keychain: &K, index: u32) -> bool {
        self.inner.unmark_used(&(keychain.clone(), index))
    }

    /// Iterates over all unused script pubkeys for a `keychain` that have been stored in the index.
    pub fn unused_scripts(&self, keychain: &K) -> impl DoubleEndedIterator<Item = (u32, &Script)> {
        let next_index = self.last_stored.get(keychain).map_or(0, |&v| v + 1);
        let range = (keychain.clone(), u32::MIN)..(keychain.clone(), next_index);
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
    pub fn last_used_index(&self, keychain: &K) -> Option<u32> {
        self.keychain_txouts(keychain).last().map(|(i, _)| i)
    }

    /// Returns the highest derivation index of each keychain that [`KeychainTxOutIndex`] has found
    /// a [`TxOut`] with it's script pubkey.
    pub fn last_used_indices(&self) -> BTreeMap<K, u32> {
        self.keychains
            .iter()
            .filter_map(|(keychain, _)| {
                self.last_used_index(keychain)
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
