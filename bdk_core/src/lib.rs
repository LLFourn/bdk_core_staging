#![no_std]
pub use alloc::boxed::Box;
pub use alloc::vec::Vec;
pub use bitcoin;
use bitcoin::{BlockHash, TxOut};
pub use miniscript;
mod descriptor_tracker;
pub use descriptor_tracker::*;
pub mod coin_select;
pub mod sign;

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
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
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

/// A Blockhash and Blockheight denoting a Checkpoint in the Blockchain.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct CheckPoint {
    pub height: u32,
    pub hash: BlockHash,
}

// When no-std use `alloc`'s Hash collections. This is activated by default
#[cfg(all(not(feature = "std"), not(feature = "hashbrown")))]
mod collections {
    #![allow(dead_code)]
    pub type HashSet<K> = alloc::collections::BTreeSet<K>;
    pub type HashMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub use alloc::collections::*;
}

// When we have std use `std`'s all collections
#[cfg(all(feature = "std", not(feature = "hashbrown")))]
mod collections {
    pub use std::collections::*;
}

// With special feature `hashbrown` use `hashbrown`'s hash collections, and else from `alloc`.
#[cfg(feature = "hashbrown")]
mod collections {
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
