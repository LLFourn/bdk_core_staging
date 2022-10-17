use bdk_core::{BlockId, CheckpointCandidate, TxGraph};
use bitcoin::{
    hashes::Hash, secp256k1::Secp256k1, BlockHash, LockTime, OutPoint, Script, Transaction, TxIn,
    TxOut, Txid,
};
use miniscript::{Descriptor, DescriptorPublicKey};

const DESCRIPTOR: &'static str = "wpkh(xpub6ERApfZwUNrhLCkDtcHTcxd75RbzS1ed54G1LkBUHQVHQKqhMkhgbmJbZRkrgZw4koxb5JaHWkY4ALHY2grBGRjaDMzQLcgJvLJuZZvRcEL)";

#[allow(unused)]
#[derive(Clone, Copy)]
pub enum OSpec {
    Mine(/* value */ u64, /* the derivation index */ usize),
    Other(/*value*/ u64),
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub enum ISpec {
    InCheckPoint(usize, u32),
    Explicit(OutPoint),
    Other,
}

#[derive(Clone)]
pub struct TxSpec {
    pub inputs: Vec<ISpec>,
    pub outputs: Vec<OSpec>,
    pub confirmed_at: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct CheckpointGen {
    pub vout_counter: u32,
    pub prev_tip: Option<BlockId>,
    pub descriptor: Descriptor<DescriptorPublicKey>,
}

impl CheckpointGen {
    pub fn new() -> Self {
        Self {
            vout_counter: 0,
            prev_tip: None,
            descriptor: DESCRIPTOR.parse().unwrap(),
        }
    }

    pub fn next_txin(&mut self) -> TxIn {
        let txin = TxIn {
            previous_output: OutPoint {
                txid: Txid::from_inner([0u8; 32]),
                vout: self.vout_counter,
            },
            ..Default::default()
        };
        self.vout_counter += 1;
        txin
    }

    pub fn create_update(
        &mut self,
        graph: &mut TxGraph,
        tx_specs: Vec<TxSpec>,
        checkpoint_height: u32,
    ) -> CheckpointCandidate {
        let secp = Secp256k1::verification_only();
        let mut txids: Vec<(Txid, Option<u32>)> = vec![];

        for tx_spec in tx_specs {
            let tx = Transaction {
                version: 1,
                lock_time: LockTime::ZERO.into(),
                input: tx_spec
                    .inputs
                    .iter()
                    .map(|ispec| match ispec {
                        ISpec::Explicit(outpoint) => TxIn {
                            previous_output: *outpoint,
                            ..Default::default()
                        },
                        ISpec::InCheckPoint(tx_index, vout) => TxIn {
                            previous_output: OutPoint {
                                txid: txids[*tx_index].0,
                                vout: *vout,
                            },
                            ..Default::default()
                        },
                        ISpec::Other => self.next_txin(),
                    })
                    .collect(),
                output: tx_spec
                    .outputs
                    .into_iter()
                    .map(|out_spec| -> TxOut {
                        match out_spec {
                            OSpec::Other(value) => TxOut {
                                value,
                                script_pubkey: Script::default(),
                            },
                            OSpec::Mine(value, index) => TxOut {
                                value,
                                script_pubkey: self
                                    .descriptor
                                    .at_derivation_index(index as u32)
                                    .derived_descriptor(&secp)
                                    .unwrap()
                                    .script_pubkey(),
                            },
                        }
                    })
                    .collect(),
            };

            txids.push((tx.txid(), tx_spec.confirmed_at));
            graph.insert_tx(&tx);
        }

        let new_tip = BlockId {
            height: checkpoint_height,
            hash: BlockHash::from_inner([0u8; 32]),
        };

        let update = CheckpointCandidate {
            txids,
            new_tip,
            invalidate: None,
            last_valid: self.prev_tip,
        };

        self.prev_tip = Some(new_tip);

        update
    }
}
