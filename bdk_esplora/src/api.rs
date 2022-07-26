//! structs from the esplora API
//!
//! see: <https://github.com/Blockstream/esplora/blob/master/API.md>
use bdk_core::{
    bitcoin::{OutPoint, Script, Transaction, TxIn, TxOut, Txid, Witness},
    BlockTime, PrevOuts,
};

#[derive(serde::Deserialize, Clone, Debug)]
pub struct PrevOut {
    pub value: u64,
    pub scriptpubkey: Script,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Vin {
    pub txid: Txid,
    pub vout: u32,
    // None if coinbase
    pub prevout: Option<PrevOut>,
    pub scriptsig: Script,
    #[serde(deserialize_with = "deserialize_witness", default)]
    pub witness: Vec<Vec<u8>>,
    pub sequence: u32,
    pub is_coinbase: bool,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Vout {
    pub value: u64,
    pub scriptpubkey: Script,
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct TxStatus {
    pub confirmed: bool,
    pub block_height: Option<u32>,
    pub block_time: Option<u32>,
}

impl TxStatus {
    pub fn to_block_time(&self) -> Option<BlockTime> {
        let height = self.block_height?;
        let time = self.block_time?;
        Some(BlockTime { height, time })
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Tx {
    pub txid: Txid,
    pub version: i32,
    pub locktime: u32,
    pub vin: Vec<Vin>,
    pub vout: Vec<Vout>,
    pub status: TxStatus,
    pub fee: u64,
}

impl Tx {
    pub fn to_tx(&self) -> Transaction {
        Transaction {
            version: self.version,
            lock_time: self.locktime,
            input: self
                .vin
                .iter()
                .cloned()
                .map(|vin| TxIn {
                    previous_output: OutPoint {
                        txid: vin.txid,
                        vout: vin.vout,
                    },
                    script_sig: vin.scriptsig,
                    sequence: vin.sequence,
                    witness: Witness::from_vec(vin.witness),
                })
                .collect(),
            output: self
                .vout
                .iter()
                .cloned()
                .map(|vout| TxOut {
                    value: vout.value,
                    script_pubkey: vout.scriptpubkey,
                })
                .collect(),
        }
    }

    pub fn confirmation_time(&self) -> Option<BlockTime> {
        match self.status {
            TxStatus {
                confirmed: true,
                block_height: Some(height),
                block_time: Some(time),
            } => Some(BlockTime { time, height }),
            _ => None,
        }
    }

    pub fn previous_outputs(&self) -> PrevOuts {
        if self.vin.len() == 1 && self.vin[0].prevout.is_none() {
            return PrevOuts::Coinbase;
        }

        PrevOuts::Spend(
            self.vin
                .iter()
                .cloned()
                .filter_map(|vin| {
                    vin.prevout.map(|po| TxOut {
                        script_pubkey: po.scriptpubkey,
                        value: po.value,
                    })
                })
                .collect(),
        )
    }
}

fn deserialize_witness<'de, D>(d: D) -> Result<Vec<Vec<u8>>, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    use bdk_core::bitcoin::hashes::hex::FromHex;
    use serde::Deserialize;
    let list = Vec::<String>::deserialize(d)?;
    list.into_iter()
        .map(|hex_str| Vec::<u8>::from_hex(&hex_str))
        .collect::<Result<Vec<Vec<u8>>, _>>()
        .map_err(serde::de::Error::custom)
}
