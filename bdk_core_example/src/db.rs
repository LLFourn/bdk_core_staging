use anyhow::Context;
use bdk_core::{
    chain_graph::ChainGraph, ChainIndex, KeychainChangeSet, KeychainScan, KeychainTracker,
};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Seek,
    path::Path,
};

use crate::Keychain;

#[derive(Debug)]
pub struct Db<I> {
    db_file: File,
    keychain_cache: BTreeMap<Keychain, u32>,
    chain_index: core::marker::PhantomData<I>,
}

impl<I> Db<I>
where
    for<'de> I: ChainIndex + Send + Sync + 'static + serde::Deserialize<'de> + serde::Serialize,
{
    pub fn load(
        db_path: &Path,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
    ) -> anyhow::Result<Self> {
        let mut db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(db_path.clone())
            .with_context(|| format!("trying to open {}", db_path.display()))?;

        loop {
            let pos = db_file.stream_position()?;

            match bincode::decode_from_std_read(&mut db_file, bincode::config::standard()) {
                Ok(bincode::serde::Compat(changeset @ KeychainChangeSet::<I, Keychain> { .. })) => {
                    tracker.derive_all_spks(changeset.keychain);
                    chain_graph.apply_changeset(&changeset.chain_graph);
                }
                Err(e) => {
                    if let bincode::error::DecodeError::Io { inner, .. } = &e {
                        // The only kind of error that we actually want to return are read failures
                        // caused by device failure etc. UnexpectedEof just menas that whatever was
                        // left after the last entry wasn't enough to be decoded -- we can just
                        // ignore it and write over it.
                        if inner.kind() != std::io::ErrorKind::UnexpectedEof {
                            return Err(e).context("IO error while reading next entry");
                        }
                    }
                    db_file.seek(std::io::SeekFrom::Start(pos))?;
                    break;
                }
            }
        }

        // we only scan for txouts we own after loading the transactions to avoid missing anything
        tracker.scan(chain_graph.graph());

        Ok(Self {
            keychain_cache: tracker.derivation_indicies(),
            db_file,
            chain_index: Default::default(),
        })
    }

    pub fn apply_wallet_scan(
        &mut self,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
        keychain_scan: KeychainScan<Keychain, I>,
    ) -> anyhow::Result<()> {
        tracker.derive_all_spks(keychain_scan.last_active_indexes);
        tracker.scan(keychain_scan.update.graph());
        let chain_changeset = chain_graph.determine_changeset(&keychain_scan.update)?;

        let changeset = KeychainChangeSet {
            chain_graph: chain_changeset.clone(),
            keychain: self.derivation_index_changes(tracker),
        };

        self.append_changeset(changeset)?;

        chain_graph.apply_changeset(&chain_changeset);
        Ok(())
    }

    fn derivation_index_changes(
        &self,
        tracker: &KeychainTracker<Keychain>,
    ) -> BTreeMap<Keychain, u32> {
        tracker
            .derivation_indicies()
            .into_iter()
            .filter(|(keychain, index)| self.keychain_cache.get(keychain) != Some(index))
            .collect()
    }

    fn append_changeset(
        &mut self,
        changeset: KeychainChangeSet<I, Keychain>,
    ) -> anyhow::Result<()> {
        if !changeset.is_empty() {
            bincode::encode_into_std_write(
                bincode::serde::Compat(&changeset),
                &mut self.db_file,
                bincode::config::standard(),
            )?;
        }

        Ok(())
    }

    pub fn apply_wallet_sync(
        &mut self,
        chain_graph: &mut ChainGraph<I>,
        tracker: &mut KeychainTracker<Keychain>,
        update: ChainGraph<I>,
    ) -> anyhow::Result<()> {
        tracker.scan(update.graph());
        let changeset = chain_graph.determine_changeset(&update)?;

        let keychain_changeset = KeychainChangeSet {
            chain_graph: changeset.clone(),
            keychain: Default::default(),
        };

        self.append_changeset(keychain_changeset)?;

        chain_graph.apply_changeset(&changeset);
        Ok(())
    }

    pub fn set_derivation_indicies(
        &mut self,
        tracker: &KeychainTracker<Keychain>,
    ) -> anyhow::Result<()> {
        let keychain_changeset = KeychainChangeSet {
            chain_graph: Default::default(),
            keychain: self.derivation_index_changes(tracker),
        };

        self.append_changeset(keychain_changeset)?;

        Ok(())
    }
}
