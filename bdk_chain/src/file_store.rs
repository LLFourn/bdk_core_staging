use crate::{
    collections::BTreeMap,
    keychain::{KeychainChangeSet, KeychainTracker},
    sparse_chain,
};
use core::marker::PhantomData;
use std::{
    fs::{File, OpenOptions},
    io::{self, Seek},
    path::Path,
};

/// Persists an append only list of `KeychainChangeSet<K,P>` to a single file.
/// [`KeychainChangeSet<K,P>`] record the changes made to a [`KeychainTracker<K,P>`].
#[derive(Debug)]
pub struct KeychainStore<K, P> {
    db_file: File,
    chain_index: core::marker::PhantomData<(K, P)>,
}

impl<K, P> KeychainStore<K, P>
where
    K: Ord + Clone + core::fmt::Debug,
    P: sparse_chain::ChainPosition,
    KeychainChangeSet<K, P>: serde::Serialize + serde::de::DeserializeOwned,
{
    /// Creates a new store from a [`File`].
    ///
    /// The file must have been opened with read, write permissions.
    ///
    /// [`File`]: std::fs::File
    pub fn new(file: File) -> Self {
        Self {
            db_file: file,
            chain_index: Default::default(),
        }
    }

    /// Creates or loads a a store from `db_path`. If no file exists there it will be created.
    pub fn new_from_path(db_path: &Path) -> Result<Self, io::Error> {
        let db_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(db_path.clone())?;

        Ok(Self::new(db_file))
    }

    /// Iterates over the stored changeset from first to last changing the seek position at each
    /// iteration.
    ///
    /// The iterator may fail to read an entry and therefore return an error. However the first time
    /// it returns an error will be the last. After doing so the iterator will always yield `None`.
    ///
    /// **WARNING**: This method changes the write position in the underlying file. You should
    /// always iterate over all entries until `None` is returned if you want your next write to go
    /// at the end, otherwise you writing over existing enties.
    pub fn iter_changesets(&mut self) -> Result<EntryIter<'_, KeychainChangeSet<K, P>>, io::Error> {
        self.db_file.rewind()?;

        Ok(EntryIter::new(&mut self.db_file))
    }

    /// Loads all the changesets that have been stored as one giant changeset.
    ///
    /// This function returns a tuple of the aggregate changeset and a result which indicates
    /// whether an error occurred while reading or deserializing one of the entries. If so the
    /// changeset will consist of all of those it was able to read.
    ///
    /// You should usually check the error. In many applications it may make sense to do a full
    /// wallet scan with a stop gap after getting an error since it is likely that one of the
    /// changesets it was unable to read changed the derivation indicies of the tracker.
    ///
    /// **WARNING**: This method changes the write position of the underlying file. The next
    /// changeset will be written over the erroring entry (or the end of the file if none existed).
    pub fn aggregate_changeset(&mut self) -> (KeychainChangeSet<K, P>, Result<(), IterError>) {
        let mut changeset = KeychainChangeSet::default();
        let result = (|| {
            let iter_changeset = self.iter_changesets()?;
            for next_changeset in iter_changeset {
                changeset.append(next_changeset?);
            }
            Ok(())
        })();

        (changeset, result)
    }

    /// Reads and applies all the changesets stored sequentially to tracker, stopping when it fails
    /// to read the next one.
    ///
    /// **WARNING**: This method changes the write position of the underlying file. The next
    /// changeset will be written over the erroring entry (or the end of the file if none existed).
    pub fn load_into_keychain_tracker(
        &mut self,
        tracker: &mut KeychainTracker<K, P>,
    ) -> Result<(), IterError> {
        for changeset in self.iter_changesets()? {
            tracker.apply_changeset(changeset?)
        }
        Ok(())
    }

    /// Append a new changeset to the file.
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
                self.db_file.sync_data()?;
            }
        }

        Ok(())
    }

    /// Appends a new changeset setting the derivation indicies
    pub fn set_derivation_indices(&mut self, indices: BTreeMap<K, u32>) -> Result<(), io::Error> {
        let keychain_changeset = KeychainChangeSet {
            chain_graph: Default::default(),
            derivation_indices: indices,
        };
        self.append_changeset(&keychain_changeset)?;

        Ok(())
    }
}

#[derive(Debug)]
pub enum IterError {
    Io(io::Error),
    Bincode(bincode::error::DecodeError),
}

impl core::fmt::Display for IterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IterError::Io(e) => write!(f, "io error trying to read entry {}", e),
            IterError::Bincode(e) => write!(f, "bincode error while reading entry {}", e),
        }
    }
}

impl std::error::Error for IterError {}

/// Iterator over entries in a file store.
///
/// Reads and returns an entry each time [`next`] is called. If an error occurs while reading the
/// iterator will yield a `Result::Err(_)` instead and then `None` for the next call to `next`.
///
/// [`next`]: Self::next
pub struct EntryIter<'a, V> {
    db_file: &'a mut File,
    types: PhantomData<V>,
    error_exit: bool,
}

impl<'a, V> EntryIter<'a, V> {
    pub fn new(db_file: &'a mut File) -> Self {
        Self {
            db_file,
            types: PhantomData,
            error_exit: false,
        }
    }
}

impl<'a, V> Iterator for EntryIter<'a, V>
where
    V: serde::de::DeserializeOwned,
{
    type Item = Result<V, IterError>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = (|| {
            let pos = self.db_file.stream_position()?;

            match bincode::decode_from_std_read(self.db_file, bincode::config::standard()) {
                Ok(bincode::serde::Compat(changeset)) => Ok(Some(changeset)),
                Err(e) => {
                    if let bincode::error::DecodeError::Io { inner, .. } = &e {
                        if inner.kind() == io::ErrorKind::UnexpectedEof {
                            let eof = self.db_file.seek(io::SeekFrom::End(0))?;
                            if pos == eof {
                                return Ok(None);
                            }
                        }
                    }

                    self.db_file.seek(io::SeekFrom::Start(pos))?;
                    Err(IterError::Bincode(e))
                }
            }
        })();

        let result = result.transpose();

        if let Some(Err(_)) = &result {
            self.error_exit = true;
        }

        result
    }
}

impl From<io::Error> for IterError {
    fn from(value: io::Error) -> Self {
        IterError::Io(value)
    }
}
