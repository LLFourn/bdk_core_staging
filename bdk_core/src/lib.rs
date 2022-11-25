#![no_std]
pub use bitcoin;
pub mod chain_graph;
mod spk_txout_index;
pub use spk_txout_index::*;
mod chain_data;
pub use chain_data::*;
pub mod coin_select;
mod for_each_txout;
pub mod keychain;
pub mod sparse_chain;
pub mod tx_graph;
pub use for_each_txout::*;

#[allow(unused_imports)]
#[macro_use]
extern crate alloc;

#[cfg(feature = "serde")]
extern crate serde_crate as serde;

#[cfg(feature = "bincode")]
extern crate bincode_crate as bincode;

#[cfg(feature = "std")]
#[macro_use]
extern crate std;

#[cfg(all(not(feature = "std"), feature = "hashbrown"))]
extern crate hashbrown;

// When no-std use `alloc`'s Hash collections. This is activated by default
#[cfg(all(not(feature = "std"), not(feature = "hashbrown")))]
#[doc(hidden)]
pub mod collections {
    #![allow(dead_code)]
    pub type HashSet<K> = alloc::collections::BTreeSet<K>;
    pub type HashMap<K, V> = alloc::collections::BTreeMap<K, V>;
    pub use alloc::collections::*;
}

// When we have std use `std`'s all collections
#[cfg(all(feature = "std", not(feature = "hashbrown")))]
#[doc(hidden)]
pub mod collections {
    pub use std::collections::*;
}

// With special feature `hashbrown` use `hashbrown`'s hash collections, and else from `alloc`.
#[cfg(feature = "hashbrown")]
#[doc(hidden)]
pub mod collections {
    #![allow(dead_code)]
    pub type HashSet<K> = hashbrown::HashSet<K>;
    pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
    pub use alloc::collections::*;
}
