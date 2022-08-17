use crate::{collections::*, SpkTracker};
use bitcoin::{secp256k1::Secp256k1, Script};
use core::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};
use miniscript::{Descriptor, DescriptorPublicKey};

/// A convienient way of tracking script pubkeys associated with one or more descriptors together.
///
/// `DeRef`s to the inner [`SpkTracker`]
///
/// [`SpkTracker`]: crate::SpkTracker
#[derive(Clone, Debug)]
pub struct KeychainTracker<K> {
    inner: SpkTracker<(K, u32)>,
    descriptors: BTreeMap<K, Descriptor<DescriptorPublicKey>>,
}

impl<K> Default for KeychainTracker<K> {
    fn default() -> Self {
        Self {
            inner: SpkTracker::default(),
            descriptors: BTreeMap::default(),
        }
    }
}

impl<K> Deref for KeychainTracker<K> {
    type Target = SpkTracker<(K, u32)>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<K> DerefMut for KeychainTracker<K> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<K: Clone + Ord + Debug> KeychainTracker<K> {
    pub fn iter_keychains(
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

    pub fn descriptor(&self, keychain: K) -> &Descriptor<DescriptorPublicKey> {
        self.descriptors
            .get(&keychain)
            .expect("keychain does not exist")
    }

    ///
    pub fn next_derivation_index(&self, keychain: K) -> u32 {
        self.inner
            .script_pubkeys()
            .range(&(keychain.clone(), u32::MIN)..=&(keychain.clone(), u32::MAX))
            .last()
            .map(|((_, last), _)| last + 1)
            .unwrap_or(0)
    }
    /// Derives script pubkeys from the descriptor **up to and including** `end` and stores them
    /// unless a script already exists in that index.
    ///
    /// Returns whether any new were derived (or if they had already all been stored).
    pub fn derive_spks(&mut self, keychain: K, end: u32) -> bool {
        let descriptor = self
            .descriptors
            .get(&keychain)
            .expect(&format!("no descriptor for keychain {:?}", keychain));
        let secp = Secp256k1::verification_only();
        let end = match descriptor.has_wildcard() {
            false => 0,
            true => end,
        };
        let next_to_derive = self.next_derivation_index(keychain.clone());
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
    /// The tracker returns a new script pubkey for each call to this method and stores it internally so
    /// it will be able to find transactions related to it.
    pub fn derive_new(&mut self, keychain: K) -> (u32, &Script) {
        let secp = Secp256k1::verification_only();
        let next_derivation_index = self.next_derivation_index(keychain.clone());
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

    pub fn derive_next_unused(&mut self, keychain: K) -> (u32, &Script) {
        let need_new = self
            .inner
            .iter_unused()
            .filter(|((kc, _), _)| kc == &keychain)
            .next()
            .is_none();
        // this rather strange branch is needed because of some lifetime issues
        if need_new {
            self.derive_new(keychain)
        } else {
            self.inner
                .iter_unused()
                .filter(|((kc, _), _)| kc == &keychain)
                .map(|((_, i), script)| (i, script))
                .next()
                .unwrap()
        }
    }

    // pub fn create_psbt(
    //     &self,
    //     inputs: impl IntoIterator<Item = (OutPoint , Option<&Plan>)>,
    //     outputs: impl IntoIterator<Item = TxOut>,
    //     chain: &SparseChain,
    // ) -> (Psbt, BTreeMap<usize, Descriptor<DefiniteDescriptorKey>>) {
    //     let unsigned_tx = Transaction {
    //         version: 0x02,
    //         lock_time: 0x00,
    //         input: inputs
    //             .into_iter()
    //             .map(|(previous_output, _)| TxIn {
    //                 previous_output,
    //                 ..Default::default()
    //             })
    //             .collect(),
    //         output: outputs.into_iter().collect(),
    //     };

    //     let mut psbt = Psbt::from_unsigned_tx(unsigned_tx).unwrap();
    //     let mut definite_descriptors = BTreeMap::new();

    //     for ((input_index, psbt_input), txin) in psbt
    //         .inputs
    //         .iter_mut()
    //         .enumerate()
    //         .zip(&psbt.unsigned_tx.input)
    //     {

    //         if let Some(definite_descriptor) = definite_descriptor {
    //             let prev_tx = chain
    //                 .get_tx(txin.previous_output.txid)
    //                 .expect("since the txout exists so must the transaction");
    //             match definite_descriptor.desc_type().segwit_version() {
    //                 Some(version) => {
    //                     if version < WitnessVersion::V1 {
    //                         psbt_input.non_witness_utxo = Some(prev_tx.tx.clone());
    //                     }
    //                     psbt_input.witness_utxo =
    //                         Some(prev_tx.tx.output[txin.previous_output.vout as usize].clone());
    //                 }
    //                 None => psbt_input.non_witness_utxo = Some(prev_tx.tx.clone()),
    //             }

    //             psbt_input
    //                 .update_with_descriptor_unchecked(&definite_descriptor)
    //                 .expect("conversion error cannot happen if descriptor is well formed");
    //             definite_descriptors.insert(input_index, definite_descriptor);
    //         }
    //     }

    //     (psbt, definite_descriptors)
    // }
}
