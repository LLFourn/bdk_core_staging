#[allow(unused_macros)]
macro_rules! h {
    ($index:literal) => {{
        bitcoin::hashes::Hash::hash($index.as_bytes())
    }};
}

#[allow(unused_macros)]
macro_rules! chain {
    ($([$($tt:tt)*]),*) => { chain!( checkpoints: [$([$($tt)*]),*] ) };
    (checkpoints: $($tail:tt)*) => { chain!( index: TxHeight, checkpoints: $($tail)*) };
    (index: $ind:ty, checkpoints: [ $([$height:expr, $block_hash:expr]),* ] $(,txids: [$(($txid:expr, $tx_height:expr)),*])?) => {{
        #[allow(unused_mut)]
        let mut chain = bdk_core::sparse_chain::SparseChain::<$ind>::from_checkpoints([$(($height, $block_hash).into()),*]);

        $(
            $(
                chain.insert_tx($txid, $tx_height).unwrap();
            )*
        )?

        chain
    }};
}

#[allow(unused_macros)]
macro_rules! changeset {
    (checkpoints: $($tail:tt)*) => { changeset!(index: TxHeight, checkpoints: $($tail)*) };
    (
        index: $ind:ty,
        checkpoints: [ $(( $height:expr, $cp_to:expr )),* ]
        $(,txids: [ $(( $txid:expr, $tx_to:expr )),* ])?
    ) => {{
        use bdk_core::collections::BTreeMap;

        #[allow(unused_mut)]
        bdk_core::sparse_chain::ChangeSet::<$ind> {
            checkpoints: {
                let mut changes = BTreeMap::default();
                $(changes.insert($height, $cp_to);)*
                changes
            },
            txids: {
                let mut changes = BTreeMap::default();
                $($(changes.insert($txid, $tx_to.map(|h: TxHeight| h.into()));)*)?
                changes
            }
        }
    }};
}
