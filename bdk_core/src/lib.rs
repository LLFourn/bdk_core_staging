#![no_std]
pub use alloc::{boxed::Box, vec::Vec};
pub use bitcoin;
use bitcoin::{hashes::Hash, BlockHash, TxOut};
mod spk_tracker;
pub use spk_tracker::*;
mod sparse_chain;
pub use sparse_chain::*;
mod tx_graph;
pub use tx_graph::*;
pub mod coin_select;
mod io;
#[cfg(feature = "miniscript")]
mod keychain_tracker;
#[cfg(feature = "miniscript")]
pub use keychain_tracker::*;
#[cfg(feature = "miniscript")]
pub use miniscript;
#[cfg(feature = "miniscript")]
mod descriptor_ext;
#[cfg(feature = "miniscript")]
pub use descriptor_ext::*;

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "serde")]
extern crate serde_crate as serde;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[cfg(all(not(feature = "std"), feature = "hashbrown"))]
extern crate hashbrown;

/// Block height and timestamp of a block
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct BlockTime {
    /// confirmation block height
    pub height: u32,
    /// confirmation block timestamp
    pub time: u64,
}

/// A reference to a block in the cannonical chain.
#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct BlockId {
    /// The height the block was confirmed at
    pub height: u32,
    /// The hash of the block
    pub hash: BlockHash,
}

impl Default for BlockId {
    fn default() -> Self {
        Self {
            height: Default::default(),
            hash: BlockHash::from_inner([0u8; 32]),
        }
    }
}

impl From<(u32, BlockHash)> for BlockId {
    fn from((height, hash): (u32, BlockHash)) -> Self {
        Self { height, hash }
    }
}

impl From<(&u32, &BlockHash)> for BlockId {
    fn from((height, hash): (&u32, &BlockHash)) -> Self {
        Self {
            height: *height,
            hash: *hash,
        }
    }
}

// When no-std use `alloc`'s Hash collections. This is activated by default
#[cfg(all(not(feature = "std"), not(feature = "hashbrown")))]
pub mod collections {
    #![allow(dead_code)]
    pub type HashSet<K> = alloc::collections::BTreeSet<K>;
    pub type HashMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub use alloc::collections::*;
}

// When we have std use `std`'s all collections
#[cfg(all(feature = "std", not(feature = "hashbrown")))]
pub mod collections {
    pub use std::collections::*;
}

// With special feature `hashbrown` use `hashbrown`'s hash collections, and else from `alloc`.
#[cfg(feature = "hashbrown")]
pub mod collections {
    #![allow(dead_code)]
    pub type HashSet<K> = hashbrown::HashSet<K>;
    pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
    pub use alloc::collections::*;
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrevOuts {
    Coinbase,
    Spend(Vec<TxOut>),
}

pub mod testing {
    use bitcoin::{hashes::Hash, Txid};

    use crate::BlockId;

    pub fn gen_hash<H: Hash>(n: u64) -> H {
        Hash::hash(&n.to_le_bytes()[..])
    }

    pub fn gen_txid(n: u64) -> Txid {
        gen_hash(n)
    }

    pub fn gen_block_id(height: u32, hash_n: u64) -> BlockId {
        BlockId {
            height,
            hash: gen_hash(hash_n),
        }
    }
}
