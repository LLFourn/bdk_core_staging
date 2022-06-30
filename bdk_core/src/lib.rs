#![no_std]
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;
pub use bitcoin;
use bitcoin::{BlockHash, TxOut};
pub use miniscript;
mod descriptor_tracker;
pub use descriptor_tracker::*;
pub mod coin_select;
pub mod sign;

#[allow(unused_imports)]
extern crate alloc;

#[cfg(feature = "serde")]
extern crate serde_crate as serde;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

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

/// Block height and timestamp of a block
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

// TODO: use the proper one if wev've got std or hashbrown if not
type HashMap<K, V> = BTreeMap<K, V>;
type HashSet<K> = BTreeSet<K>;

#[derive(Clone, Debug, PartialEq)]
pub enum PrevOuts {
    Coinbase,
    Spend(Vec<TxOut>),
}
