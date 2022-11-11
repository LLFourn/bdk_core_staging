use alloc::vec::Vec;
use core::mem::size_of;
use std::io::{self, Read, Seek, SeekFrom, Write};

/// A very simple collection of `256` byte linked lists.
#[derive(Debug)]
pub struct ByteList<S> {
    inner: S,
}

/// The number of magic bytes
const MAGIC_BYTES_SIZE: usize = 4;

/// The magic bytes we expect to read at the start of the file
const MAGIC_BYTES: [u8; MAGIC_BYTES_SIZE] = [0x26, 0xd3, 0x00, 0x00];

/// The tail table is a fixed size lookup for each tail of the linked list
const TAIL_TABLE_SIZE: usize = size_of::<u64>() * u8::MAX as usize;

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    UnexpectedVersion,
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LoadError::Io(e) => e.fmt(f),
            LoadError::UnexpectedVersion => {
                write!(f, "file did not begin with the expected magic bytes")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LoadError {}

impl From<io::Error> for LoadError {
    fn from(value: io::Error) -> Self {
        LoadError::Io(value)
    }
}

macro_rules! read_ints {
    ($reader:expr => $($type:ty),*) => {{
        let mut main_buf = [0u8; 0 $(+size_of::<$type>())*];
        $reader.read_exact(&mut main_buf)?;
        let mut _pos = 0;
        ($({
            let mut buf = [0u8; size_of::<$type>()];
            buf.copy_from_slice(&main_buf[_pos..(_pos + size_of::<$type>())]);
            _pos += size_of::<$type>();
            <$type>::from_be_bytes(buf)
        }),*)
    }}
}

impl<S: Read + Write + Seek> ByteList<S> {
    /// Load the byte source ready for Read/Write/Seek
    pub fn load(mut stream: S) -> Result<Self, LoadError> {
        stream.rewind()?;
        let mut type_version = [0u8; 4];
        if let Err(e) = stream.read_exact(&mut type_version) {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                stream.rewind()?;
                stream.write_all(&MAGIC_BYTES)?;
                stream.write_all(&[0u8; TAIL_TABLE_SIZE])?;
                type_version = MAGIC_BYTES;
            } else {
                return Err(LoadError::Io(e));
            }
        }

        if type_version != MAGIC_BYTES {
            return Err(LoadError::UnexpectedVersion);
        }

        Ok(Self { inner: stream })
    }

    /// Initialize stream with [`MAGIC_BYTES`] and linked list 
    /// tail lookup table
    pub fn init(mut stream: S) -> Result<Self, io::Error> {
        stream.rewind()?;
        stream.write_all(&MAGIC_BYTES[..])?;
        stream.write_all(&[0u8; TAIL_TABLE_SIZE])?;

        Ok(Self { inner: stream })
    }

    /// Use key (object type) to push new serialized changeset to the byte linked list
    pub fn push(&mut self, key: u8, data: &[u8]) -> io::Result<()> {
        assert!(data.len() <= u32::MAX as usize);
        let mut header = [0u8; 12];
        let tail_loc = self.tail_loc(key)?;
        let new_tail_loc = self.inner.seek(SeekFrom::End(0))?;
        header[0..8].copy_from_slice(tail_loc.to_be_bytes().as_ref());
        header[8..12].copy_from_slice((data.len() as u32).to_be_bytes().as_ref());
        // Write new entry first so we're sure it's written before we update pointer to tail
        self.inner.write_all(&header)?;
        self.inner.write_all(data)?;
        self.inner.flush()?;
        // it worked so let's set the pointer
        self.set_tail_loc(key, new_tail_loc)?;
        Ok(())
    }

    /// Get the last item in the byte linked list
    pub fn last(&mut self, key: u8) -> io::Result<Option<Vec<u8>>> {
        self.iter(key)?.next().transpose()
    }

    /// Push bytes if last changeset is different from incoming changeset
    pub fn push_if_different(&mut self, key: u8, data: &[u8]) -> io::Result<()> {
        if self.last(key)?.as_ref().map(|x| &x[..]) != Some(data) {
            self.push(key, data)?;
        }
        Ok(())
    }

    /// Iterates from the last element to the first
    pub fn iter(&mut self, key: u8) -> io::Result<IterEntries<'_, S>> {
        let last_entry_pos = self.tail_loc(key)?;
        Ok(IterEntries {
            inner: &mut self.inner,
            current: last_entry_pos,
        })
    }

    /// Get all the keys stored in the byte linked list
    pub fn keys(&mut self) -> io::Result<impl DoubleEndedIterator<Item = u8>> {
        self.inner.seek(SeekFrom::Start(MAGIC_BYTES_SIZE as u64))?;
        let mut key_index = [0u8; TAIL_TABLE_SIZE];
        self.inner.read_exact(&mut key_index)?;
        Ok((0..u8::MAX).filter(move |key| {
            let start = *key as usize * size_of::<u64>();
            let end = start + size_of::<u64>();
            &key_index[start..end] != &[0u8; size_of::<u64>()]
        }))
    }

    /// Move the cursor to the position of the last entry
    fn seek_to_last_entry_pos(&mut self, key: u8) -> io::Result<()> {
        let last_entry_loc =
            SeekFrom::Start(key as u64 * size_of::<u64>() as u64 + MAGIC_BYTES_SIZE as u64);
        self.inner.seek(last_entry_loc)?;
        Ok(())
    }

    /// Get the tail location for a particular key (object type)
    fn tail_loc(&mut self, key: u8) -> io::Result<u64> {
        self.seek_to_last_entry_pos(key)?;
        Ok(read_ints!(self.inner => u64))
    }

    /// Set the tail location for a particular key (object type)
    fn set_tail_loc(&mut self, key: u8, pos: u64) -> io::Result<()> {
        self.seek_to_last_entry_pos(key)?;
        self.inner.write_all(pos.to_be_bytes().as_ref())?;
        self.inner.flush()?;
        Ok(())
    }
}

/// Iterator over entries of the byte linked list
pub struct IterEntries<'a, S> {
    inner: &'a mut S,
    current: u64,
}

impl<'a, S: Read + Write + Seek> Iterator for IterEntries<'a, S> {
    type Item = Result<Vec<u8>, io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current == 0x00 {
            return None;
        }

        let result = (|| {
            self.inner.seek(SeekFrom::Start(self.current))?;
            let (next_loc, entry_len) = read_ints!(self.inner => u64 , u32);

            self.current = next_loc;

            let mut entry = vec![0u8; entry_len as usize];
            self.inner.read_exact(&mut entry)?;
            Ok(entry)
        })();

        Some(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use alloc::collections::LinkedList;
    use proptest::test_runner::TestRng;
    use proptest::{prelude::*, test_runner::RngAlgorithm};
    use rand::{Rng, RngCore};

    proptest! {

        #![proptest_config(ProptestConfig {
            timeout: 100,
            .. ProptestConfig::default()
        })]
        #[test]
        fn fail_after_fill_up_buffer(size in 0usize..1_000_000) {
            dbg!(size);
            let mut buf = vec![0u8;size];
            let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
            let mut bytelist = match ByteList::init(io::Cursor::new(&mut buf[..])) {
                Ok(bytelist) => bytelist,
                Err(_e) => {
                    assert!(size < MAGIC_BYTES_SIZE + TAIL_TABLE_SIZE);
                    return Ok(());
                },
            };

            let mut in_memory = (0..=u8::MAX).map(|_| LinkedList::new()).collect::<Vec<_>>();

            loop {
                let key = rng.gen_range(0..=u8::MAX);
                let byte_len = rng.gen_range(0..1_000);
                let mut bytes = vec![0u8;byte_len];
                rng.fill_bytes(&mut bytes);
                match bytelist.push(key, &bytes) {
                    Ok(_) => in_memory[key as usize].push_back(bytes),
                    Err(_) => {
                        break
                    }
                }
            }

            let mut bytelist = ByteList::load(io::Cursor::new(&mut buf[..])).unwrap();

            for (key, list) in in_memory.into_iter().enumerate() {
                let truth = list.into_iter().rev().collect::<Vec<_>>();
                let our_iter = bytelist.iter(key as u8).unwrap().collect::<Result<Vec<_>,_>>().unwrap();
                assert_eq!(truth, our_iter);
            }
        }
    }

    #[test]
    fn write_some_enties() {
        let a1 = b"a1".to_vec();
        let a2 = b"a2".to_vec();
        let b1 = b"b1".to_vec();
        let target: io::Cursor<Vec<u8>> = io::Cursor::new(vec![]);
        let mut byte_list = ByteList::load(target).unwrap();

        assert_eq!(byte_list.iter(0).unwrap().count(), 0);

        byte_list.push(0, &a1).unwrap();
        assert_eq!(byte_list.iter(0).unwrap().count(), 1);
        assert_eq!(byte_list.keys().unwrap().collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            byte_list
                .iter(0)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![a1.clone()]
        );
        byte_list.push(0, &a2).unwrap();
        assert_eq!(
            byte_list
                .iter(0)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![a2.clone(), a1.clone()]
        );
        byte_list.push(1, &b1).unwrap();
        assert_eq!(byte_list.keys().unwrap().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(
            byte_list
                .iter(0)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![a2.clone(), a1.clone()]
        );
        assert_eq!(
            byte_list
                .iter(1)
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
            vec![b1.clone()]
        );
    }
}
