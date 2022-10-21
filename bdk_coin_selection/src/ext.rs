use crate::varint_size;
use core::mem::size_of;

const TXOUT_BASE_WEIGHT: u32 = 4 * size_of::<u64>() as u32; // just the value

pub trait TxOutExt {
    fn weight(&self) -> u32;
}

impl TxOutExt for bitcoin::TxOut {
    fn weight(&self) -> u32 {
        let spk_len = self.script_pubkey.len();
        TXOUT_BASE_WEIGHT + (varint_size(spk_len) + spk_len as u32) * 4
    }
}
