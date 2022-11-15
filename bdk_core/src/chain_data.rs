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

impl ChainIndex for TxHeight {
    fn height(&self) -> TxHeight {
        *self
    }

    fn max_ord_of_height(height: TxHeight) -> Self {
        height
    }

    fn min_ord_of_height(height: TxHeight) -> Self {
        height
    }
}

impl TxHeight {
    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed(_))
    }
}

/// Block height and timestamp in which a transaction is confirmed in.
#[derive(Debug, Clone, PartialEq, Eq, Copy, PartialOrd, Ord, core::hash::Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub enum ConfirmationTime {
    Confirmed { height: u32, time: u64 },
    Unconfirmed,
}

impl ChainIndex for ConfirmationTime {
    fn height(&self) -> TxHeight {
        match self {
            ConfirmationTime::Confirmed { height, .. } => TxHeight::Confirmed(*height),
            ConfirmationTime::Unconfirmed => TxHeight::Unconfirmed,
        }
    }

    fn max_ord_of_height(height: TxHeight) -> Self {
        match height {
            TxHeight::Confirmed(height) => Self::Confirmed {
                height,
                time: u64::MAX,
            },
            TxHeight::Unconfirmed => Self::Unconfirmed,
        }
    }

    fn min_ord_of_height(height: TxHeight) -> Self {
        match height {
            TxHeight::Confirmed(height) => Self::Confirmed {
                height,
                time: u64::MIN,
            },
            TxHeight::Unconfirmed => Self::Unconfirmed,
        }
    }
}

impl ConfirmationTime {
    pub fn is_confirmed(&self) -> bool {
        matches!(self, Self::Confirmed { .. })
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
pub struct FullTxOut<I> {
    pub outpoint: OutPoint,
    pub txout: TxOut,
    pub chain_index: I,
    pub spent_by: Option<Txid>,
}

/// Represents an index in which transactions are ordered by in [`SparseChain`].
///
/// [`ChainIndex`] implementations must be [`Ord`] by [`TxHeight`] first.
pub trait ChainIndex:
    core::fmt::Debug + Clone + Copy + Eq + PartialOrd + Ord + core::hash::Hash
{
    /// Obtain the transaction height of the index.
    fn height(&self) -> TxHeight;

    /// Obtain the index's upper bound of a given height.
    fn max_ord_of_height(height: TxHeight) -> Self;

    /// Obtain the index's lower bound of a given height.
    fn min_ord_of_height(height: TxHeight) -> Self;
}

#[cfg(test)]
pub mod verify_chain_index {
    use alloc::vec::Vec;

    use super::ChainIndex;
    use crate::{ConfirmationTime, TxHeight};

    pub fn verify_chain_index<I: ChainIndex>(head_count: u32, tail_count: u32) {
        let values = (0..head_count)
            .chain(u32::MAX - tail_count..u32::MAX)
            .flat_map(|i| {
                [
                    I::min_ord_of_height(TxHeight::Confirmed(i)),
                    I::max_ord_of_height(TxHeight::Confirmed(i)),
                ]
            })
            .chain([
                I::min_ord_of_height(TxHeight::Unconfirmed),
                I::max_ord_of_height(TxHeight::Unconfirmed),
            ])
            .collect::<Vec<_>>();

        for i in 0..values.len() {
            for j in 0..values.len() {
                if i == j {
                    assert_eq!(values[i], values[j]);
                }
                if i < j {
                    assert!(values[i] <= values[j]);
                }
                if i > j {
                    assert!(values[i] >= values[j]);
                }
            }
        }
    }

    #[test]
    fn verify_tx_height() {
        verify_chain_index::<TxHeight>(1000, 1000);
    }

    #[test]
    fn verify_confirmation_time() {
        verify_chain_index::<ConfirmationTime>(1000, 1000);
    }
}
