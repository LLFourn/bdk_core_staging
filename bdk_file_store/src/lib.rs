use anyhow::Context;
use bdk_chain::{keychain::KeychainChangeSet, keychain_tracker::KeychainTracker, sparse_chain};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::Seek,
    path::Path,
};

/// Persists changes made to a [`KeychainTracker`] to a file so they can be restored later on.
#[derive(Debug)]
pub struct KeychainStore<K, I> {
    db_file: File,
    /// A cache of what the current state of the derivation indexes are on-disk
    deriviation_index_cache: BTreeMap<K, u32>,
    chain_index: core::marker::PhantomData<(K, I)>,
}

impl<K, I> KeychainStore<K, I>
where
    K: Ord + Clone + core::fmt::Debug,
    I: sparse_chain::ChainPosition,
    KeychainChangeSet<K, I>: serde::Serialize + serde::de::DeserializeOwned,
{
    pub fn load(db_path: &Path, tracker: &mut KeychainTracker<K, I>) -> anyhow::Result<Self> {
        let mut db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(db_path.clone())
            .with_context(|| format!("trying to open {}", db_path.display()))?;

        loop {
            let pos = db_file.stream_position()?;

            let failed =
                match bincode::decode_from_std_read(&mut db_file, bincode::config::standard()) {
                    Ok(bincode::serde::Compat(changeset @ KeychainChangeSet::<K, I> { .. })) => {
                        tracker
                            .txout_index
                            .store_all_up_to(&changeset.derivation_indices);
                        tracker.apply_changeset(changeset).is_err()
                    }
                    Err(e) => {
                        if let bincode::error::DecodeError::Io { inner, .. } = &e {
                            // The only kind of error that we actually want to return are read failures
                            // caused by device failure etc. UnexpectedEof just menas that whatever was
                            // left after the last entry wasn't enough to be decoded (usually its 0
                            // bytes) -- If it's not empty we can just ignore it and write over the
                            // corrupted entry.
                            if inner.kind() != std::io::ErrorKind::UnexpectedEof {
                                return Err(e).context("IO error while reading next entry");
                            }
                        }
                        true
                    }
                };

            if failed {
                db_file.seek(std::io::SeekFrom::Start(pos))?;
                break;
            }
        }

        Ok(Self {
            deriviation_index_cache: tracker.txout_index.derivation_indices(),
            db_file,
            chain_index: Default::default(),
        })
    }

    pub fn append_changeset(&mut self, changeset: &KeychainChangeSet<K, I>) -> anyhow::Result<()> {
        if !changeset.is_empty() {
            bincode::encode_into_std_write(
                bincode::serde::Compat(changeset),
                &mut self.db_file,
                bincode::config::standard(),
            )?;

            // We want to make sure that derivation indexe changes are written to disk as soon as
            // possible so you know about the write failure before you give ou the address in the application.
            if !changeset.derivation_indices.is_empty() {
                self.db_file.sync_data()?
            }
        }

        Ok(())
    }

    pub fn set_derivation_indices(&mut self, indices: BTreeMap<K, u32>) -> anyhow::Result<()> {
        let keychain_changeset = KeychainChangeSet {
            chain_graph: Default::default(),
            derivation_indices: indices
                .into_iter()
                .filter(|(keychain, index)| {
                    self.deriviation_index_cache.get(keychain) != Some(index)
                })
                .collect(),
        };

        self.append_changeset(&keychain_changeset)?;

        Ok(())
    }
}
