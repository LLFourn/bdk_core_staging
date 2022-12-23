use crate::{
    chain_graph::{self, ChainGraph},
    collections::BTreeMap,
    tx_graph::TxGraph,
    ForEachTxout,
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
pub struct KeychainChangeSet<K, I> {
    /// The changes in local keychain derivation indices
    pub derivation_indices: BTreeMap<K, u32>,
    /// The changes that have occurred in the blockchain
    pub chain_graph: chain_graph::ChangeSet<I>,
}

impl<K, I> Default for KeychainChangeSet<K, I> {
    fn default() -> Self {
        Self {
            chain_graph: Default::default(),
            derivation_indices: Default::default(),
        }
    }
}

impl<K, I> KeychainChangeSet<K, I> {
    pub fn is_empty(&self) -> bool {
        self.chain_graph.is_empty() && self.derivation_indices.is_empty()
    }
}

impl<K, I> From<chain_graph::ChangeSet<I>> for KeychainChangeSet<K, I> {
    fn from(changeset: chain_graph::ChangeSet<I>) -> Self {
        Self {
            chain_graph: changeset,
            ..Default::default()
        }
    }
}

impl<K, I> AsRef<TxGraph> for KeychainScan<K, I> {
    fn as_ref(&self) -> &TxGraph {
        self.update.graph()
    }
}

impl<K, I> ForEachTxout for KeychainChangeSet<K, I> {
    fn for_each_txout(&self, f: &mut impl FnMut((bitcoin::OutPoint, &bitcoin::TxOut))) {
        self.chain_graph.for_each_txout(f)
    }
}
