use bdk_chain::SpkTxOutIndex;
use bitcoin::{hashes::hex::FromHex, OutPoint, PackedLockTime, Script, Transaction, TxIn, TxOut};

#[test]
fn spk_txout_sent_and_received() {
    let spk1 = Script::from_hex("001404f1e52ce2bab3423c6a8c63b7cd730d8f12542c").unwrap();
    let spk2 = Script::from_hex("00142b57404ae14f08c3a0c903feb2af7830605eb00f").unwrap();

    let mut index = SpkTxOutIndex::default();
    index.add_spk(0, spk1.clone());
    index.add_spk(1, spk2.clone());

    let tx1 = Transaction {
        version: 0x02,
        lock_time: PackedLockTime(0),
        input: vec![],
        output: vec![TxOut {
            value: 42_000,
            script_pubkey: spk1.clone(),
        }],
    };

    assert_eq!(index.sent_and_received(&tx1), (0, 42_000));
    assert_eq!(index.net_value(&tx1), 42_000);
    index.scan(&tx1);
    assert_eq!(
        index.sent_and_received(&tx1),
        (0, 42_000),
        "shouldn't change after scanning"
    );

    let tx2 = Transaction {
        version: 0x1,
        lock_time: PackedLockTime(0),
        input: vec![TxIn {
            previous_output: OutPoint {
                txid: tx1.txid(),
                vout: 0,
            },
            ..Default::default()
        }],
        output: vec![
            TxOut {
                value: 20_000,
                script_pubkey: spk2.clone(),
            },
            TxOut {
                script_pubkey: spk1.clone(),
                value: 30_000,
            },
        ],
    };

    assert_eq!(index.sent_and_received(&tx2), (42_000, 50_000));
    assert_eq!(index.net_value(&tx2), 8_000);
}
