#[no_std]
#[allow(unused)]
#[macro_use]
extern crate alloc;

mod coin_selector;
pub use coin_selector::*;

// mod bnb;
// pub use bnb::*;
mod bnb;
pub mod metrics;

mod feerate;
pub use feerate::*;
pub mod change_policy;
pub mod ext;

/// Txin "base" fields include `outpoint` (32+4) and `nSequence` (4). This does not include
/// `scriptSigLen` or `scriptSig`.
pub const TXIN_BASE_WEIGHT: u32 = (32 + 4 + 4) * 4;

/// Helper to calculate varint size. `v` is the value the varint represents.
// Shamelessly copied from
// https://github.com/rust-bitcoin/rust-miniscript/blob/d5615acda1a7fdc4041a11c1736af139b8c7ebe8/src/util.rs#L8
pub(crate) fn varint_size(v: usize) -> u32 {
    bitcoin::VarInt(v as u64).len() as u32
}
