//! structs from the esplora API
//!
//! see: <https://github.com/Blockstream/esplora/blob/master/API.md>
use bdk_core::{
    bitcoin::{
        BlockHash, OutPoint, PackedLockTime, Script, Sequence, Transaction, TxIn, TxOut, Txid,
        Witness,
    },
    BlockId, ConfirmationTime, PrevOuts, Timestamp,
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
    pub sequence: Sequence,
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
    pub block_time: Option<u64>,
}

impl TxStatus {
    pub fn into_confirmation_time(self, fallback_time: u64) -> ConfirmationTime {
        ConfirmationTime {
            height: self.block_height.into(),
            time: Timestamp(self.block_time.unwrap_or(fallback_time)),
        }
    }
}

#[derive(serde::Deserialize, Clone, Debug)]
pub struct Tx {
    pub txid: Txid,
    pub version: i32,
    pub locktime: PackedLockTime,
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

    pub fn confirmation_time(&self, fallback_time: u64) -> ConfirmationTime {
        self.status.clone().into_confirmation_time(fallback_time)
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

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Block {
    pub id: BlockHash,
    pub height: u32,
}

impl Block {
    pub fn block_id(&self) -> BlockId {
        BlockId {
            hash: self.id,
            height: self.height,
        }
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
