use crate::chain_graph::ChainGraph;
use crate::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
/// An update that includes the last active indexes of each keychain.
pub struct WalletScanUpdate<K, I> {
    /// The update data in the form of a chain that could be applied
    pub update: ChainGraph<I>,
    /// The last active indexes of each keychain
    pub last_active_indexes: BTreeMap<K, u32>,
}

impl<K,I> Default for WalletScanUpdate<K,I> {
    fn default() -> Self {
        Self {
            update: Default::default(),
            last_active_indexes: Default::default(),
        }
    }
}
