use crate::{
    collections::HashSet,
    sparse_chain::{self, ChainPosition, SparseChain},
    tx_graph::{self, TxGraph},
    BlockId, ForEachTxout, FullTxOut, TxHeight,
};
use alloc::{string::ToString, vec::Vec};
use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

/// A convenient combination of a [`SparseChain<P>`] and a [`TxGraph`].
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
pub struct ChainGraph<P = TxHeight> {
    chain: SparseChain<P>,
    graph: TxGraph,
}

impl<P> Default for ChainGraph<P> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<P> ChainGraph<P> {
    pub fn chain(&self) -> &sparse_chain::SparseChain<P> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }
}

impl<P: ChainPosition> ChainGraph<P> {
    pub fn checkpoint_limit(&self) -> Option<usize> {
        self.chain.checkpoint_limit()
    }

    pub fn set_checkpoint_limit(&mut self, limit: Option<usize>) {
        self.chain.set_checkpoint_limit(limit)
    }

    /// Get a transaction that is currently in the underlying [`SparseChain`]. This doesn't
    /// necessarily mean that it is *confirmed* in the blockchain, it might just be in the
    /// unconfirmed transaction list within the [`SparseChain`].
    pub fn get_tx_in_chain(&self, txid: Txid) -> Option<(&P, &Transaction)> {
        let position = self.chain.tx_position(txid)?;
        let full_tx = self.graph.get_tx(txid).expect("must exist");
        Some((position, full_tx))
    }

    /// Determines the changes required to insert a transaction into the inner [`ChainGraph`] and
    /// [`SparseChain`] at the given `position`.
    ///
    /// If inserting it into the chain `position` will result in conflicts, the returned
    /// [`ChangeSet`] should evict conflicting transactions.
    pub fn insert_tx_preview(
        &self,
        tx: Transaction,
        pos: P,
    ) -> Result<ChangeSet<P>, InsertTxFailure<P>> {
        // only allow displacement of unconfirmed txs
        let chain_changeset = self.chain.insert_tx_preview(tx.txid(), pos)?;

        Ok(self
            .inflate_changeset(chain_changeset, core::iter::once(tx))
            .map_err(|failure| match failure {
                InflateError::Missing(_) => unreachable!("only one tx added and we provided it"),
                InflateError::UnresolvableConflict(conflict) => {
                    InsertTxFailure::UnresolvableConflict(conflict)
                }
            })?)
    }

    /// Inserts [`Transaction`] at given chain position. This is equivalent to calling
    /// [`Self::insert_tx_preview()`] and [`Self::apply_changeset()`] in sequence.
    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        pos: P,
    ) -> Result<ChangeSet<P>, InsertTxFailure<P>> {
        let changeset = self.insert_tx_preview(tx, pos)?;
        self.apply_changeset(changeset.clone())
            .expect("changeset should not have missing transactions");
        Ok(changeset)
    }

    /// Determines the changes required to insert a [`TxOut`] into the internal [`TxGraph`].
    pub fn insert_txout_preview(&self, outpoint: OutPoint, txout: TxOut) -> ChangeSet<P> {
        ChangeSet {
            chain: Default::default(),
            graph: self.graph.insert_txout_preview(outpoint, txout),
        }
    }

    /// Inserts a [`TxOut`] into the internal [`TxGraph`]. This is equivalent to calling
    /// [`Self::insert_txout_preview()`] and [`Self::apply_changeset`] in sequence.
    pub fn insert_txout(&mut self, outpoint: OutPoint, txout: TxOut) -> ChangeSet<P> {
        let changeset = self.insert_txout_preview(outpoint, txout);
        self.apply_changeset(changeset.clone())
            .expect("changeset should not have missing transactions");
        changeset
    }

    /// Determines the changes required to insert a `block_id` (a height and block hash) into the
    /// chain.
    ///
    /// If a checkpoint already exists at that height with a different hash this will return
    /// an error.
    pub fn insert_checkpoint_preview(
        &self,
        block_id: BlockId,
    ) -> Result<ChangeSet<P>, InsertCheckpointFailure> {
        self.chain
            .insert_checkpoint_preview(block_id)
            .map(|chain_changeset| ChangeSet {
                chain: chain_changeset,
                ..Default::default()
            })
    }

    /// Inserts checkpoint into [`Self`]. This is equivilant to calling
    /// [`Self::insert_checkpoint_preview()`] and [`Self::apply_changeset()] in sequence.
    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<ChangeSet<P>, InsertCheckpointFailure> {
        let changeset = self.insert_checkpoint_preview(block_id)?;
        self.apply_changeset(changeset.clone())
            .expect("changeset should not have missing transactions");
        Ok(changeset)
    }

    /// Calculates the difference between self and `update` in the form of a [`ChangeSet`].
    pub fn determine_changeset(&self, update: &Self) -> Result<ChangeSet<P>, UpdateFailure<P>> {
        let chain_changeset = self
            .chain
            .determine_changeset(&update.chain)
            .map_err(UpdateFailure::Chain)?;

        let mut changeset = ChangeSet::<P> {
            chain: chain_changeset,
            graph: self.graph.determine_additions(&update.graph),
        };

        self.fix_conflicts(&mut changeset)?;
        Ok(changeset)
    }

    /// Given a transaction, return an iterator of in-chain [`Txid`]s that conflict with it (spends
    /// at least one of the same inputs).
    ///
    /// This method is comparable to [`TxGraph::conflicting_txids()`] which returns all conflicting
    /// transactions, whereas this method only returns conflicts that exist in the [`SparseChain`].
    pub fn conflicting_txids_in_chain<'a>(
        &'a self,
        tx: &'a Transaction,
    ) -> impl Iterator<Item = (&'a P, Txid)> + 'a {
        self.graph
            .conflicting_txids(tx)
            .filter_map(|(_, conflicting_txid)| {
                self.chain
                    .tx_position(conflicting_txid)
                    .map(|conflicting_pos| (conflicting_pos, conflicting_txid))
            })
    }

    /// Fix changeset conflicts.
    ///
    /// **WARNING:** If there are any missing full txs, conflict resolution will not be complete. In
    /// debug mode, this will result in panic.
    fn fix_conflicts(&self, changeset: &mut ChangeSet<P>) -> Result<(), UnresolvableConflict<P>> {
        let chain_conflicts = changeset
            .chain
            .txids
            .iter()
            // txid is not already in chain
            .filter(|(&txid, _)| self.chain.tx_position(txid).is_none())
            // change is add to pos
            .filter_map(|(&txid, pos_change)| pos_change.as_ref().map(|pos| (txid, pos)))
            // ensure full tx exists (either in graph, or additions)
            .filter_map(|(txid, pos)| {
                let full_tx = self
                    .graph
                    .get_tx(txid)
                    .or_else(|| changeset.graph.tx.iter().find(|tx| tx.txid() == txid))
                    .map(|tx| (txid, tx, pos));
                debug_assert!(full_tx.is_some(), "should have full tx at this point");
                full_tx
            })
            .flat_map(|(new_txid, new_tx, new_pos)| {
                self.conflicting_txids_in_chain(new_tx)
                    .map(move |(conflict_pos, conflict_txid)| {
                        (new_pos.clone(), new_txid, conflict_pos, conflict_txid)
                    })
            })
            .collect::<Vec<_>>();

        for (update_pos, update_txid, conflicting_pos, conflicting_txid) in chain_conflicts {
            // We have found a tx that conflicts with our update txid. Only allow this when the
            // conflicting tx will be positioned as "unconfirmed" after the update is applied.
            // If so, we will modify the changeset to evict the conflicting txid.

            // determine the position of the conflicting txid after current changeset is applied
            let conflicting_new_pos = changeset
                .chain
                .txids
                .get(&conflicting_txid)
                .map(Option::as_ref)
                .unwrap_or(Some(conflicting_pos));

            match conflicting_new_pos {
                None => {
                    // conflicting txid will be deleted, can ignore
                }
                Some(existing_new_pos) => match existing_new_pos.height() {
                    TxHeight::Confirmed(_) => {
                        // the new postion of the conflicting tx is "confirmed", therefore cannot be
                        // evicted, return error
                        return Err(UnresolvableConflict {
                            already_confirmed_tx: (conflicting_pos.clone(), conflicting_txid),
                            update_tx: (update_pos.clone(), update_txid),
                        });
                    }
                    TxHeight::Unconfirmed => {
                        // the new position of the conflicting tx is "unconfirmed", therefore it can
                        // be evicted
                        changeset.chain.txids.insert(conflicting_txid, None);
                    }
                },
            };
        }

        Ok(())
    }

    /// Applies [`ChangeSet`] to [`Self`]. This fails if there are missing full transactions.
    pub fn apply_changeset(&mut self, changeset: ChangeSet<P>) -> Result<(), InflateError<P>> {
        let mut missing: HashSet<Txid> = self.chain.changeset_additions(&changeset.chain).collect();

        for tx in &changeset.graph.tx {
            missing.remove(&tx.txid());
        }

        missing.retain(|txid| self.graph.get_tx(*txid).is_none());

        if missing.is_empty() {
            self.chain.apply_changeset(changeset.chain);
            self.graph.apply_additions(changeset.graph);
            Ok(())
        } else {
            Err(InflateError::Missing(missing))
        }
    }

    /// Convets a [`sparse_chain::ChangeSet`] to a valid [`ChangeSet`] by providing
    /// full transactions for each addition.
    pub fn inflate_changeset(
        &self,
        changeset: sparse_chain::ChangeSet<P>,
        full_txs: impl IntoIterator<Item = Transaction>,
    ) -> Result<ChangeSet<P>, InflateError<P>> {
        // need to wrap in a refcell because it's closed over twice below
        let missing = core::cell::RefCell::new(
            self.chain
                .changeset_additions(&changeset)
                .filter(|txid| self.graph.get_tx(*txid).is_none())
                .collect::<HashSet<_>>(),
        );
        let full_txs = full_txs
            .into_iter()
            .take_while(|_| !missing.borrow().is_empty())
            .filter(|tx| missing.borrow_mut().remove(&tx.txid()))
            .collect();

        let missing = missing.into_inner();

        if missing.is_empty() {
            let mut changeset = ChangeSet {
                chain: changeset,
                graph: tx_graph::Additions {
                    tx: full_txs,
                    ..Default::default()
                },
            };
            self.fix_conflicts(&mut changeset)?;
            Ok(changeset)
        } else {
            Err(InflateError::Missing(missing))
        }
    }

    /// Applies the `update` chain graph. Note this is shorthand for calling
    /// [`Self::determine_changeset()`] and [`Self::apply_changeset()`] in sequence.
    pub fn apply_update(&mut self, update: Self) -> Result<ChangeSet<P>, UpdateFailure<P>> {
        let changeset = self.determine_changeset(&update)?;
        self.apply_changeset(changeset.clone())
            .expect("we correctly constructed this");
        Ok(changeset)
    }

    /// Get the full transaction output at an outpoint if it exists in the chain and the graph.
    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut<P>> {
        self.chain.full_txout(&self.graph, outpoint)
    }

    /// Iterate over the full transactions and their position in the chain ordered by their position
    /// in ascending order.
    pub fn transactions_in_chain(&self) -> impl DoubleEndedIterator<Item = (&P, &Transaction)> {
        self.chain
            .txids()
            .map(|(pos, txid)| (pos, self.graph.get_tx(*txid).expect("must exist")))
    }

    /// Finds the transaction in the chain that spends `outpoint` given the input/output
    /// relationships in `graph`. Note that the transaction including `outpoint` does not need to be
    /// in the `graph` or the `chain` for this to return `Some(_)`.
    pub fn spent_by(&self, outpoint: OutPoint) -> Option<(&P, Txid)> {
        self.chain.spent_by(&self.graph, outpoint)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(crate = "serde_crate")
)]
#[must_use]
pub struct ChangeSet<P> {
    pub chain: sparse_chain::ChangeSet<P>,
    pub graph: tx_graph::Additions,
}

impl<P> ChangeSet<P> {
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty() && self.graph.is_empty()
    }

    pub fn contains_eviction(&self) -> bool {
        self.chain
            .txids
            .iter()
            .any(|(_, new_pos)| new_pos.is_none())
    }
}

impl<P> Default for ChangeSet<P> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<P> AsRef<TxGraph> for ChainGraph<P> {
    fn as_ref(&self) -> &TxGraph {
        &self.graph
    }
}

impl<P> ForEachTxout for ChangeSet<P> {
    fn for_each_txout(&self, f: &mut impl FnMut((OutPoint, &TxOut))) {
        self.graph.for_each_txout(f)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InsertTxFailure<P> {
    Chain(sparse_chain::InsertTxFailure<P>),
    UnresolvableConflict(UnresolvableConflict<P>),
}

impl<P: core::fmt::Debug> core::fmt::Display for InsertTxFailure<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InsertTxFailure::Chain(inner) => core::fmt::Display::fmt(inner, f),
            InsertTxFailure::UnresolvableConflict(inner) => core::fmt::Display::fmt(inner, f),
        }
    }
}

impl<P> From<sparse_chain::InsertTxFailure<P>> for InsertTxFailure<P> {
    fn from(inner: sparse_chain::InsertTxFailure<P>) -> Self {
        Self::Chain(inner)
    }
}

#[cfg(feature = "std")]
impl<P: core::fmt::Debug> std::error::Error for InsertTxFailure<P> {}

pub type InsertCheckpointFailure = sparse_chain::InsertCheckpointFailure;

/// Represents an update failure.
#[derive(Clone, Debug, PartialEq)]
pub enum UpdateFailure<P> {
    /// The update chain was inconsistent with the existing chain
    Chain(sparse_chain::UpdateFailure<P>),
    /// A transaction in the update spent the same input as an already confirmed transaction
    UnresolvableConflict(UnresolvableConflict<P>),
}

impl<P: core::fmt::Debug> core::fmt::Display for UpdateFailure<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            UpdateFailure::Chain(inner) => core::fmt::Display::fmt(inner, f),
            UpdateFailure::UnresolvableConflict(inner) => core::fmt::Display::fmt(inner, f),
        }
    }
}

impl<P> From<sparse_chain::UpdateFailure<P>> for UpdateFailure<P> {
    fn from(inner: sparse_chain::UpdateFailure<P>) -> Self {
        Self::Chain(inner)
    }
}

#[cfg(feature = "std")]
impl<P: core::fmt::Debug> std::error::Error for UpdateFailure<P> {}

/// Represents a failure that occured when attempting to [apply] or [inflate] a [`ChangeSet`]
///
/// [inflate]: ChainGraph::inflate_changeset
/// [apply]: ChainGraph::apply_changeset
#[derive(Clone, Debug, PartialEq)]
pub enum InflateError<P> {
    /// Missing full transactions
    Missing(HashSet<Txid>),
    /// A transaction in the update spent the same input as an already confirmed transaction
    UnresolvableConflict(UnresolvableConflict<P>),
}

impl<P: core::fmt::Debug> core::fmt::Display for InflateError<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InflateError::Missing(missing) => write!(
                f,
                "missing full transactions for {}",
                missing
                    .into_iter()
                    .map(|txid| txid.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            InflateError::UnresolvableConflict(inner) => {
                write!(f, "{}", inner)
            }
        }
    }
}

#[cfg(feature = "std")]
impl<P: core::fmt::Debug> std::error::Error for InflateError<P> {}

/// Represents an unresolvable conflict between an update's transaction and an
/// already-confirmed transaction.
#[derive(Clone, Debug, PartialEq)]
pub struct UnresolvableConflict<P> {
    pub already_confirmed_tx: (P, Txid),
    pub update_tx: (P, Txid),
}

impl<P: core::fmt::Debug> core::fmt::Display for UnresolvableConflict<P> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            already_confirmed_tx,
            update_tx,
        } = self;
        write!(f, "update transaction {} at height {:?} conflicts with an already confirmed transaction {} at height {:?}", 
            update_tx.1, update_tx.0, already_confirmed_tx.1, already_confirmed_tx.0)
    }
}

impl<P> From<UnresolvableConflict<P>> for UpdateFailure<P> {
    fn from(inner: UnresolvableConflict<P>) -> Self {
        Self::UnresolvableConflict(inner)
    }
}

impl<P> From<UnresolvableConflict<P>> for InflateError<P> {
    fn from(inner: UnresolvableConflict<P>) -> Self {
        Self::UnresolvableConflict(inner)
    }
}

impl<P> From<UnresolvableConflict<P>> for InsertTxFailure<P> {
    fn from(inner: UnresolvableConflict<P>) -> Self {
        Self::UnresolvableConflict(inner)
    }
}

#[cfg(feature = "std")]
impl<P: core::fmt::Debug> std::error::Error for UnresolvableConflict<P> {}
