use crate::Box;
use bitcoin::{secp256k1::Secp256k1, Script};
use miniscript::{Descriptor, DescriptorPublicKey};

pub trait DescriptorExt {
    /// Iterates over all the script pubkeys of a descriptor.
    fn iter_all_scripts(&self) -> Box<dyn Iterator<Item = Script> + Send>;
    fn dust_value(&self) -> u64;
}

impl DescriptorExt for Descriptor<DescriptorPublicKey> {
    fn iter_all_scripts(&self) -> Box<dyn Iterator<Item = Script> + Send> {
        let descriptor = self.clone();
        let secp = Secp256k1::verification_only();
        let end = if self.has_wildcard() {
            // Because we only iterate over non-hardened indexes there are 2^31 values
            (1 << 31) - 1
        } else {
            0
        };

        Box::new((0..=end).map(move |i| {
            descriptor
                .at_derivation_index(i)
                .derived_descriptor(&secp)
                .expect("the descritpor cannot need hardened derivation")
                .script_pubkey()
        }))
    }

    fn dust_value(&self) -> u64 {
        self.at_derivation_index(0)
            .script_pubkey()
            .dust_value()
            .to_sat()
    }
}
