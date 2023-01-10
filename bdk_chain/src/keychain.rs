//! Modules for keychain based structures.
//!
//! A keychain here is a set of application defined indexes for a minscript descriptor where we can
//! derive script pubkeys at a particular derivation index. The application's index is simply
//! anything that implemetns `Ord`.
use crate::{
    chain_graph::{self, ChainGraph},
    collections::BTreeMap,
    tx_graph::TxGraph,
    ForEachTxout,
};

#[cfg(feature = "miniscript")]
mod keychain_tracker;
#[cfg(feature = "miniscript")]
pub use keychain_tracker::*;
#[cfg(feature = "miniscript")]
mod keychain_txout_index;
#[cfg(feature = "miniscript")]
pub use keychain_txout_index::*;

#[derive(Clone, Debug, PartialEq)]
/// An update that includes the last active indexes of each keychain.
pub struct KeychainScan<K, P> {
    /// The update data in the form of a chain that could be applied
    pub update: ChainGraph<P>,
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
            deserialize = "K: Ord + serde::Deserialize<'de>, P: serde::Deserialize<'de>",
            serialize = "K: Ord + serde::Serialize, P: serde::Serialize"
        )
    )
)]
#[must_use]
pub struct KeychainChangeSet<K, P> {
    /// The changes in local keychain derivation indices
    pub derivation_indices: BTreeMap<K, u32>,
    /// The changes that have occurred in the blockchain
    pub chain_graph: chain_graph::ChangeSet<P>,
}

impl<K, P> Default for KeychainChangeSet<K, P> {
    fn default() -> Self {
        Self {
            chain_graph: Default::default(),
            derivation_indices: Default::default(),
        }
    }
}

impl<K, P> KeychainChangeSet<K, P> {
    pub fn is_empty(&self) -> bool {
        self.chain_graph.is_empty() && self.derivation_indices.is_empty()
    }
}

impl<K, P> From<chain_graph::ChangeSet<P>> for KeychainChangeSet<K, P> {
    fn from(changeset: chain_graph::ChangeSet<P>) -> Self {
        Self {
            chain_graph: changeset,
            ..Default::default()
        }
    }
}

impl<K, P> AsRef<TxGraph> for KeychainScan<K, P> {
    fn as_ref(&self) -> &TxGraph {
        self.update.graph()
    }
}

impl<K, P> ForEachTxout for KeychainChangeSet<K, P> {
    fn for_each_txout(&self, f: &mut impl FnMut((bitcoin::OutPoint, &bitcoin::TxOut))) {
        self.chain_graph.for_each_txout(f)
    }
}
