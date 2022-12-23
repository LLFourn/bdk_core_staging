#![no_std]

#[cfg(feature = "std")]
extern crate std;
#[macro_use]
extern crate alloc;

#[cfg(feature = "serde")]
extern crate serde_crate as serde;

mod keychain_txout_index;
pub use keychain_txout_index::*;
mod keychain_tracker;
pub use keychain_tracker::*;
mod descriptor_ext;
pub use bdk_core::{self, keychain::*, miniscript};
pub use descriptor_ext::*;
