use crate::{
    chain_graph::{self, ChainGraph},
    collections::BTreeMap,
};

#[derive(Clone, Debug, PartialEq)]
/// An update that includes the last active indexes of each keychain.
pub struct KeychainScan<K, I> {
    /// The update data in the form of a chain that could be applied
    pub update: ChainGraph<I>,
    /// The last active indexes of each keychain
    pub last_active_indexes: BTreeMap<K, u32>,
}

impl<K, I> Default for KeychainScan<K, I> {
    fn default() -> Self {
        Self {
            update: Default::default(),
            last_active_indexes: Default::default(),
        }
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(
        crate = "serde_crate",
        bound(
            deserialize = "K: Ord + serde::Deserialize<'de>, I: serde::Deserialize<'de>",
            serialize = "K: Ord + serde::Serialize, I: serde::Serialize"
        )
    )
)]
pub struct KeychainChangeSet<I, K> {
    /// The changes that have occured in the blockchain
    pub chain_graph: chain_graph::ChangeSet<I>,
    /// The changes in local keychain derivation indicies
    pub keychain: BTreeMap<K, u32>,
}

impl<I, K> Default for KeychainChangeSet<I, K> {
    fn default() -> Self {
        Self {
            chain_graph: Default::default(),
            keychain: Default::default(),
        }
    }
}

impl<I, K> KeychainChangeSet<I, K> {
    pub fn is_empty(&self) -> bool {
        self.chain_graph.is_empty() && self.keychain.is_empty()
    }
}
