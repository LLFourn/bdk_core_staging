use crate::sparse_chain::{self, ChainIndex, ChainIndexExtension};
use bitcoin::{hashes::Hash, BlockHash, OutPoint, TxOut, Txid};

/// Represents the height in which a transaction is confirmed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub enum TxHeight {
    Confirmed(u32),
    Unconfirmed,
}

impl Default for TxHeight {
    fn default() -> Self {
        Self::Unconfirmed
    }
}

impl core::fmt::Display for TxHeight {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Confirmed(h) => core::write!(f, "confirmed_at({})", h),
            Self::Unconfirmed => core::write!(f, "unconfirmed"),
        }
    }
}

impl From<Option<u32>> for TxHeight {
    fn from(opt: Option<u32>) -> Self {
        match opt {
            Some(h) => Self::Confirmed(h),
            None => Self::Unconfirmed,
        }
    }
}

impl From<TxHeight> for Option<u32> {
    fn from(height: TxHeight) -> Self {
        match height {
            TxHeight::Confirmed(h) => Some(h),
            TxHeight::Unconfirmed => None,
        }
    }
}

impl TxHeight {
    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }
}

/// Block height and timestamp in which a transaction is confirmed in.
#[derive(Debug, Clone, PartialEq, Eq, Default, Copy, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct ConfirmationTime {
    pub height: TxHeight,
    pub time: Option<u64>,
}

impl ConfirmationTime {
    pub fn is_confirmed(&self) -> bool {
        self.height.is_confirmed()
    }
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

impl From<BlockId> for (u32, BlockHash) {
    fn from(block_id: BlockId) -> Self {
        (block_id.height, block_id.hash)
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

/// A `TxOut` with as much data as we can retreive about it
#[derive(Debug, Clone, PartialEq)]
pub struct FullTxOut<E> {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub chain_index: sparse_chain::ChainIndex<E>,
    pub spent_by: Option<Txid>,
}

/// A wrapped `u64` for use as a [`ChainIndexExtension`](crate::sparse_chain::ChainIndexExtension)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Hash)]
pub struct Timestamp(pub u64);

impl ChainIndexExtension for Timestamp {
    const MIN: Self = Timestamp(u64::MIN);
    const MAX: Self = Timestamp(u64::MAX);
}

impl From<ConfirmationTime> for ChainIndex<Option<Timestamp>> {
    fn from(ct: ConfirmationTime) -> Self {
        ChainIndex {
            height: ct.height,
            extension: ct.time.map(Timestamp),
        }
    }
}
