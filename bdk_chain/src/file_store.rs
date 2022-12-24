use crate::{
    collections::BTreeMap,
    keychain::{KeychainChangeSet, KeychainTracker},
    sparse_chain,
};
use std::{
    fs::{File, OpenOptions},
    io::{self, Seek},
    path::Path,
};

/// Persists changes made to a [`KeychainTracker`] to a file so they can be restored later on.
#[derive(Debug)]
pub struct KeychainStore<K, P> {
    db_file: File,
    /// A cache of what the current state of the derivation indexes are on-disk
    deriviation_index_cache: BTreeMap<K, u32>,
    chain_index: core::marker::PhantomData<(K, P)>,
}

impl<K, P> KeychainStore<K, P>
where
    K: Ord + Clone + core::fmt::Debug,
    P: sparse_chain::ChainPosition,
    KeychainChangeSet<K, P>: serde::Serialize + serde::de::DeserializeOwned,
{
    pub fn load(db_path: &Path, tracker: &mut KeychainTracker<K, P>) -> Result<Self, io::Error> {
        let mut db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(db_path.clone())?;

        loop {
            let pos = db_file.stream_position()?;

            let failed =
                match bincode::decode_from_std_read(&mut db_file, bincode::config::standard()) {
                    Ok(bincode::serde::Compat(changeset @ KeychainChangeSet::<K, P> { .. })) => {
                        tracker
                            .txout_index
                            .store_all_up_to(&changeset.derivation_indices);
                        tracker.apply_changeset(changeset).is_err()
                    }
                    Err(e) => {
                        if let bincode::error::DecodeError::Io { inner, .. } = e {
                            // The only kind of error that we actually want to return are read failures
                            // caused by device failure etc. UnexpectedEof just menas that whatever was
                            // left after the last entry wasn't enough to be decoded (usually its 0
                            // bytes) -- If it's not empty we can just ignore it and write over the
                            // corrupted entry.
                            if inner.kind() != io::ErrorKind::UnexpectedEof {
                                return Err(inner);
                            }
                        }
                        true
                    }
                };

            if failed {
                db_file.seek(io::SeekFrom::Start(pos))?;
                break;
            }
        }

        Ok(Self {
            deriviation_index_cache: tracker.txout_index.derivation_indices(),
            db_file,
            chain_index: Default::default(),
        })
    }

    pub fn append_changeset(
        &mut self,
        changeset: &KeychainChangeSet<K, P>,
    ) -> Result<(), io::Error> {
        if !changeset.is_empty() {
            bincode::encode_into_std_write(
                bincode::serde::Compat(changeset),
                &mut self.db_file,
                bincode::config::standard(),
            )
            .map_err(|e| match e {
                bincode::error::EncodeError::Io { inner, .. } => inner,
                unexpected_err => panic!("unexpected bincode error: {}", unexpected_err),
            })?;

            // We want to make sure that derivation indexe changes are written to disk as soon as
            // possible so you know about the write failure before you give ou the address in the application.
            if !changeset.derivation_indices.is_empty() {
                self.db_file.sync_data()?
            }
        }

        Ok(())
    }

    pub fn set_derivation_indices(&mut self, indices: BTreeMap<K, u32>) -> Result<(), io::Error> {
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
