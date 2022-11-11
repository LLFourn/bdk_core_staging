use bdk_core::chain_graph;
use bdk_core::collections::BTreeMap;
use bdk_core::disklist::{DiskList, LoadError};
use bincode;
use std::io;

use crate::Keychain;

#[derive(Clone, Debug, Copy)]
enum DbKey {
    ChainData,
    DerivationIndex,
}

pub struct Db<F>(DiskList<F>);

const BINCODE_CONF: bincode::config::Configuration = bincode::config::standard();

impl<F: io::Read + io::Write + io::Seek> Db<F> {
    pub fn load(file: F) -> Result<Self, LoadError> {
        Ok(Self(DiskList::load(file)?))
    }

    pub fn push_chain_changeset(&mut self, changeset: &chain_graph::ChangeSet) -> io::Result<()> {
        let buf = bincode::encode_to_vec(
            bincode::serde::Compat(changeset),
            bincode::config::standard(),
        )
        .unwrap();
        self.0.push(DbKey::ChainData as u8, &buf[..])
    }

    pub fn iter_changesets(
        &mut self,
    ) -> io::Result<impl DoubleEndedIterator<Item = chain_graph::ChangeSet>> {
        Ok(self
            .0
            .iter(DbKey::ChainData as u8)?
            .map(|bytes| {
                bincode::decode_from_slice::<bincode::serde::Compat<chain_graph::ChangeSet>, _>(
                    &bytes?,
                    BINCODE_CONF,
                )
                .map(|x| x.0 .0)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            })
            .collect::<Result<Vec<chain_graph::ChangeSet>, _>>()?
            .into_iter()
            .rev())
    }

    pub fn get_derivation_indicies(&mut self) -> io::Result<BTreeMap<Keychain, u32>> {
        Ok(self
            .0
            .last(DbKey::DerivationIndex as u8)?
            .map(|bytes| -> io::Result<BTreeMap<Keychain, u32>> {
                bincode::decode_from_slice(&bytes, BINCODE_CONF)
                    .map(|x| x.0)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            })
            .transpose()?
            .unwrap_or_else(BTreeMap::<Keychain, u32>::new))
    }

    pub fn set_derivation_indicies(&mut self, indicies: BTreeMap<Keychain, u32>) -> io::Result<()> {
        let buf = bincode::encode_to_vec(indicies, bincode::config::standard()).unwrap();
        self.0
            .push_if_different(DbKey::DerivationIndex as u8, &buf[..])
    }
}
