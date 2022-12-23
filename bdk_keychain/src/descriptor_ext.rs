use crate::miniscript::{Descriptor, DescriptorPublicKey};

pub trait DescriptorExt {
    /// Iterates over all the script pubkeys of a descriptor.
    fn dust_value(&self) -> u64;
}

impl DescriptorExt for Descriptor<DescriptorPublicKey> {
    fn dust_value(&self) -> u64 {
        self.at_derivation_index(0)
            .script_pubkey()
            .dust_value()
            .to_sat()
    }
}
