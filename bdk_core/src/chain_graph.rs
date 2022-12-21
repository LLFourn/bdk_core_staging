use crate::{
    collections::HashSet,
    sparse_chain::{self, ChainIndex, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ForEachTxout, FullTxOut, TxHeight,
};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

/// A convenient combination of a [`SparseChain<I>`] and a [`TxGraph`].
///
/// Very often you want to store transaction data when you record a transaction's existence. Adding
/// a transaction to a `ChainGraph` atomically stores the `txid` in its `SparseChain<I>`
/// while also storing the transaction data in its `TxGraph`.
///
/// The `ChainGraph` does not guarantee any 1:1 mapping between transactions in the `chain` and
/// `graph` or vis versa. Both fields are public so they can mutated indepdendly. Even if you only
/// modify the `ChainGraph` through its atomic API, keep in mind that `TxGraph` does not allow
/// deletions while `SparseChain` does so deleting a transaction from the chain cannot delete it
/// from the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainGraph<I = TxHeight> {
    chain: SparseChain<I>,
    graph: TxGraph,
}

impl<I> Default for ChainGraph<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I> ChainGraph<I> {
    pub fn chain(&self) -> &sparse_chain::SparseChain<I> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }
}

impl<I: ChainIndex> ChainGraph<I> {
    pub fn checkpoint_limit(&self) -> Option<usize> {
        self.chain.checkpoint_limit()
    }

    pub fn set_checkpoint_limit(&mut self, limit: Option<usize>) {
        self.chain.set_checkpoint_limit(limit)
    }

    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        index: Option<I>,
    ) -> Result<InsertOk, sparse_chain::InsertTxErr> {
        let chain_changed = match index {
            Some(index) => self.chain.insert_tx(tx.txid(), index)?,
            None => false,
        };
        let graph_changed = self.graph.insert_tx(tx);

        Ok(InsertOk {
            chain: chain_changed,
            graph: graph_changed,
        })
    }

    pub fn insert_output(&mut self, outpoint: OutPoint, txout: TxOut) -> bool {
        self.graph.insert_txout(outpoint, txout)
    }

    /// Insert a `block_id` (a height and block hash) into the chain. If a checkpoint already exists
    /// at that height with a different hash this will return an error. Otherwise it will return
    /// `Ok(true)` if the checkpoint didn't already exist or `Ok(false)` if it did.
    ///
    /// **Warning**: This function modifies the internal state of the chain graph. You are
    /// responsible for persisting these changes to disk if you need to restore them.
    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<bool, sparse_chain::InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    /// Calculates the difference between self and `update` in the form of a [`ChangeSet`].
    pub fn determine_changeset(&self, update: &Self) -> Result<ChangeSet<I>, UpdateFailure<I>> {
        let mut chain_changeset = self
            .chain
            .determine_changeset(&update.chain)
            .map_err(UpdateFailure::Chain)?;

        let chain_conflicts = update
            .chain
            .iter_txids()
            // skip txids that already exist in the original chain (for efficiency)
            .filter(|&(_, txid)| self.chain.tx_index(*txid).is_none())
            // choose original txs that conflicts with the update
            .flat_map(|(update_ind, update_txid)| {
                let update_tx = update.graph.tx(*update_txid).expect("must exist");
                self.graph
                    .conflicting_txids(update_tx)
                    .filter_map(move |(_, conflicting_txid)| {
                        self.chain
                            .tx_index(conflicting_txid)
                            .map(|conflicting_ind| {
                                (update_ind, *update_txid, conflicting_ind, conflicting_txid)
                            })
                    })
            });

        for (update_ind, update_txid, conflicting_ind, conflicting_txid) in chain_conflicts {
            // We have found a tx that conflicts with our update txid. Only allow this when the
            // conflicting tx will have an index that is "unconfirmed" after the update is applied.
            // If so, we will modify the changeset to evict the conflicting txid.

            // determine the index of the conflicting txid after current changeset is applied
            let conflicting_new_ind = chain_changeset
                .txids
                .get(&conflicting_txid)
                .map(Option::as_ref)
                .unwrap_or(Some(conflicting_ind));

            match conflicting_new_ind {
                None => {
                    // conflicting txid will be deleted, can ignore
                }
                Some(existing_new_index) => match existing_new_index.height() {
                    TxHeight::Confirmed(_) => {
                        // the new index of the conflicting tx is "confirmed", therefore cannot be
                        // evicted, return error
                        return Err(UpdateFailure::<I>::Conflict {
                            already_confirmed_tx: (conflicting_ind.clone(), conflicting_txid),
                            update_tx: (update_ind.clone(), update_txid),
                        });
                    }
                    TxHeight::Unconfirmed => {
                        // the new index  of the conflicting tx is "unconfirmed", therefore it can
                        // be evicted
                        chain_changeset.txids.insert(conflicting_txid, None);
                    }
                },
            };
        }

        Ok(ChangeSet::<I> {
            chain: chain_changeset,
            graph: self.graph.determine_additions(&update.graph),
        })
    }

    /// Applies a [`ChangeSet`] to the chain graph
    pub fn apply_changeset(
        &mut self,
        changeset: ChangeSet<I>,
    ) -> Result<(), (ChangeSet<I>, HashSet<Txid>)> {
        let mut missing: HashSet<Txid> = self.chain.changeset_additions(&changeset.chain).collect();

        for tx in &changeset.graph.tx {
            missing.remove(&tx.txid());
        }

        missing.retain(|txid| !self.graph.contains_txid(*txid));

        if missing.is_empty() {
            self.chain.apply_changeset(changeset.chain);
            self.graph.apply_additions(changeset.graph);
            Ok(())
        } else {
            Err((changeset, missing))
        }
    }

    /// Convets a [`sparse_chain::ChangeSet`] to a valid [`ChangeSet`] by providing
    /// full transactions for each addition.
    ///
    pub fn inflate_changeset(
        &self,
        changeset: sparse_chain::ChangeSet<I>,
        full_txs: impl IntoIterator<Item = Transaction>,
    ) -> Result<ChangeSet<I>, (sparse_chain::ChangeSet<I>, HashSet<Txid>)> {
        let mut missing = self
            .chain
            .changeset_additions(&changeset)
            .collect::<HashSet<_>>();
        missing.retain(|txid| !self.graph.contains_txid(*txid));
        // need to wrap in a refcell because it's closed over twice below
        let missing = core::cell::RefCell::new(missing);
        let full_txs = full_txs
            .into_iter()
            .take_while(|_| !missing.borrow().is_empty())
            .filter(|tx| missing.borrow_mut().remove(&tx.txid()))
            .collect();

        let missing = missing.into_inner();

        if missing.is_empty() {
            Ok(ChangeSet {
                chain: changeset,
                graph: tx_graph::Additions {
                    tx: full_txs,
                    ..Default::default()
                },
            })
        } else {
            Err((changeset, missing))
        }
    }

    /// Applies the `update` chain graph. Note this is shorthand for calling [`determine_changeset`]
    /// and [`apply_changeset`] in sequence.
    ///
    /// [`apply_changeset`]: Self::apply_changeset
    /// [`determine_changeset`]: Self::determine_changeset
    pub fn apply_update(&mut self, update: Self) -> Result<(), UpdateFailure<I>> {
        let changeset = self.determine_changeset(&update)?;
        self.apply_changeset(changeset)
            .expect("we correctly constructed this");
        Ok(())
    }

    /// Get the full transaction output at an outpoint if it exists in the chain and the graph.
    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        self.chain.full_txout(&self.graph, outpoint)
    }

    /// Finds the transaction in the chain that spends `outpoint` given the input/output
    /// relationships in `graph`. Note that the transaction including `outpoint` does not need to be
    /// in the `graph` or the `chain` for this to return `Some(_)`.
    pub fn spent_by(&self, outpoint: OutPoint) -> Option<(&I, Txid)> {
        self.chain.spent_by(&self.graph, outpoint)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
pub struct ChangeSet<I> {
    pub chain: sparse_chain::ChangeSet<I>,
    pub graph: tx_graph::Additions,
}

impl<I> ChangeSet<I> {
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty() && self.graph.is_empty()
    }
}

impl<I> Default for ChangeSet<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I> AsRef<TxGraph> for ChainGraph<I> {
    fn as_ref(&self) -> &TxGraph {
        &self.graph
    }
}

impl<I> ForEachTxout for ChangeSet<I> {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.graph.for_each_txout(f)
    }
}

pub struct InsertOk<R = bool> {
    pub chain: R,
    pub graph: R,
}

#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure<I> {
    /// The update chain was inconsistent with the existing chain
    Chain(sparse_chain::UpdateFailure<I>),
    /// A transaction in the update spent the same input as an already confirmed transaction.
    Conflict {
        already_confirmed_tx: (I, Txid),
        update_tx: (I, Txid),
    },
}

impl<I: core::fmt::Debug> core::fmt::Display for UpdateFailure<I> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UpdateFailure::Chain(chain_failure) => write!(f, "{}", chain_failure),
            UpdateFailure::Conflict {
                already_confirmed_tx,
                update_tx: new_tx,
            } => {
                write!(f, "new transaction {} at height {:?} conflicts with already confirmed transaction {} at height {:?}", new_tx.1, new_tx.0, already_confirmed_tx.1, already_confirmed_tx.0)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<I: core::fmt::Debug> std::error::Error for UpdateFailure<I> {}
