use anyhow::Context;
use bdk_core::{
    chain_graph::{self, ChainGraph},
    ChainIndex, KeychainTracker, WalletScanUpdate,
};
use std::{
    fs::{File, OpenOptions},
    io::Seek,
    path::Path,
};

use crate::Keychain;

#[derive(Debug)]
pub struct Db<I> {
    chain_db: File,
    keychain_db: File,
    chain_index: core::marker::PhantomData<I>,
}

impl<I> Db<I>
where
    I: ChainIndex + Send + Sync + 'static,
    bincode::serde::Compat<chain_graph::ChangeSet<I>>: bincode::Decode,
    for<'a> bincode::serde::Compat<&'a chain_graph::ChangeSet<I>>: bincode::Encode,
{
    pub fn load(
        db_dir: &Path,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
    ) -> anyhow::Result<Self> {
        std::fs::create_dir_all(db_dir)?;
        let chain_db_file = db_dir.join("chain.db");
        let mut chain_db = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(chain_db_file.clone())
            .with_context(|| format!("trying to open {}", chain_db_file.display()))?;

        let keychain_db_file = db_dir.join("keychain.db");
        let mut keychain_db = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(keychain_db_file.clone())
            .with_context(|| format!("trying to open {}", keychain_db_file.display()))?;

        if let Ok(derivation_indicies) =
            bincode::decode_from_std_read(&mut keychain_db, bincode::config::standard())
        {
            tracker.derive_all_spks(derivation_indicies);
        }

        while let Ok(bincode::serde::Compat(changeset)) =
            bincode::decode_from_std_read(&mut chain_db, bincode::config::standard())
        {
            chain_graph.apply_changeset(&changeset);
        }

        tracker.scan(chain_graph.graph());

        Ok(Self {
            keychain_db,
            chain_db,
            chain_index: Default::default(),
        })
    }

    pub fn apply_wallet_scan(
        &mut self,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
        wallet_scan: WalletScanUpdate<Keychain, I>,
    ) -> anyhow::Result<()> {
        tracker.derive_all_spks(wallet_scan.last_active_indexes);
        self.set_derivation_indicies(tracker)?;
        self.apply_wallet_sync(chain_graph, tracker, wallet_scan.update)?;

        Ok(())
    }

    pub fn apply_wallet_sync(
        &mut self,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
        update: ChainGraph<I>,
    ) -> anyhow::Result<()> {
        let changeset = chain_graph.determine_changeset(&update)?;
        tracker.scan(update.graph());

        bincode::encode_into_std_write(
            bincode::serde::Compat(&changeset),
            &mut self.chain_db,
            bincode::config::standard(),
        )?;

        chain_graph.apply_changeset(&changeset);
        Ok(())
    }

    pub fn set_derivation_indicies(
        &mut self,
        tracker: &KeychainTracker<Keychain>,
    ) -> anyhow::Result<()> {
        self.keychain_db.rewind()?;

        bincode::encode_into_std_write(
            tracker.derivation_indicies(),
            &mut self.keychain_db,
            bincode::config::standard(),
        )?;

        Ok(())
    }
}
