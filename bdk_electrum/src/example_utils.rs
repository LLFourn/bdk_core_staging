use bdk_chain::{
    bitcoin::secp256k1::Secp256k1,
    miniscript::{Descriptor, DescriptorPublicKey},
};
use electrsd::{
    bitcoind::{self, BitcoinD},
    ElectrsD,
};

pub struct Environment {
    // [TODO] @evanlinjin: This is temporary, before will fill out `Environment` with useful methods.
    #[allow(unused)]
    bitcoind: BitcoinD,
    electrs: ElectrsD,
}

impl Environment {
    pub fn new() -> Self {
        let bitcoind_exe = bitcoind::downloaded_exe_path().expect("bitcoind binary must exist");
        let electrs_exe = electrsd::downloaded_exe_path().expect("electrs binary must exist");
        let bitcoind = BitcoinD::new(bitcoind_exe).expect("bitcoind must run");
        let electrs = ElectrsD::new(electrs_exe, &bitcoind).expect("electrs must run");
        Self { bitcoind, electrs }
    }

    pub fn electrum_url(&self) -> &str {
        self.electrs.electrum_url.as_str()
    }
}

pub fn new_external_descriptor() -> Descriptor<DescriptorPublicKey> {
    let (descriptor, _) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&Secp256k1::signing_only(), "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)").expect("must parse");
    descriptor
}

pub fn new_internal_descriptor() -> Descriptor<DescriptorPublicKey> {
    let (descriptor, _) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&Secp256k1::signing_only(), "tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)").expect("must parse");
    descriptor
}
