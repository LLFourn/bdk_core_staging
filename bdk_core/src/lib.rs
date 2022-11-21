#![no_std]
pub use alloc::{boxed::Box, vec::Vec};
pub use bitcoin;
use bitcoin::TxOut;
pub mod chain_graph;
mod spk_tracker;
pub use spk_tracker::*;
mod chain_data;
pub use chain_data::*;
pub mod coin_select;
mod keychain;
pub use keychain::*;
#[cfg(feature = "miniscript")]
mod keychain_tracker;
pub mod sparse_chain;
pub mod tx_graph;
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
    pub use core::ops::Bound;
}

#[derive(Clone, Debug, PartialEq)]
pub enum PrevOuts {
    Coinbase,
    Spend(Vec<TxOut>),
}
