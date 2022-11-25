use anyhow::Context;
use bdk_keychain::{
    bdk_core::{keychain::KeychainChangeSet, ChainIndex},
    KeychainTracker,
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
        tracker: &mut KeychainTracker<Keychain, I>,
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
                Ok(bincode::serde::Compat(changeset @ KeychainChangeSet::<Keychain, I> { .. })) => {
                    tracker
                        .txout_index
                        .store_all_up_to(&changeset.derivation_indices);
                    tracker.apply_changeset(changeset);
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

        Ok(Self {
            keychain_cache: tracker.txout_index.derivation_indices(),
            db_file,
            chain_index: Default::default(),
        })
    }

    pub fn append_changeset(
        &mut self,
        changeset: &KeychainChangeSet<Keychain, I>,
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

    pub fn set_derivation_indices(
        &mut self,
        indices: BTreeMap<Keychain, u32>,
    ) -> anyhow::Result<()> {
        let keychain_changeset = KeychainChangeSet {
            chain_graph: Default::default(),
            derivation_indices: indices
                .into_iter()
                .filter(|(keychain, index)| self.keychain_cache.get(keychain) != Some(index))
                .collect(),
        };

        self.append_changeset(&keychain_changeset)?;

        Ok(())
    }
}
