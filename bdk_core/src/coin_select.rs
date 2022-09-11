use crate::{collections::BTreeSet, Vec};
use bitcoin::{LockTime, Transaction, TxOut};

pub const TXIN_BASE_WEIGHT: u32 = (32 + 4 + 4) * 4;

/// An [`InputCandidate`] is used as a candidate for coin selection.
#[derive(Debug, Clone, Copy)]
pub struct InputCandidate {
    /// Number of inputs contained within this [`InputCandidate`].
    /// If we are using single UTXOs as candidates, this value will be 1.
    /// If we are working with `OutputGroup`s (as done in Bitcoin Core), this would be > 1.
    pub input_count: usize,
    /// The number of input(s) contained within this candidate that is spending a segwit output.
    pub segwit_count: usize,
    /// Total value of this candidate.
    pub value: u64,
    /// Total weight of this candidate, inclusive of `prevout`, `nSequence`, `scriptSig` and
    /// `scriptWitness` fields of each input contained.
    pub weight: u32,
}

impl InputCandidate {
    /// Create a new [`InputCandidate`] that represents a single input.
    pub fn new_single(value: u64, weight: u32, is_segwit: bool) -> Self {
        Self {
            input_count: 1,
            segwit_count: if is_segwit { 1 } else { 0 },
            value,
            weight,
        }
    }

    /// This should only be used internally.
    fn empty() -> Self {
        Self {
            input_count: 0,
            segwit_count: 0,
            value: 0,
            weight: 0,
        }
    }
}

impl core::ops::AddAssign for InputCandidate {
    fn add_assign(&mut self, rhs: Self) {
        self.input_count += rhs.input_count;
        self.segwit_count += rhs.segwit_count;
        self.value += rhs.value;
        self.weight += rhs.weight;
    }
}

impl core::ops::SubAssign for InputCandidate {
    fn sub_assign(&mut self, rhs: Self) {
        self.input_count -= rhs.input_count;
        self.segwit_count -= rhs.segwit_count;
        self.value -= rhs.value;
        self.weight -= rhs.weight;
    }
}

#[derive(Debug, Clone)]
pub struct CoinSelector {
    candidates: Vec<InputCandidate>,
    selected: BTreeSet<usize>,
    opts: CoinSelectorOpt,
}

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// Ths sum of the recipient output values (in satoshis).
    pub recipients_sum: u64,

    /// The feerate we should try and achieve (in satoshis/wu).
    pub effective_feerate: f32,
    /// The long-term feerate, if we are to spend an UTXO in the future instead of now (in sats/wu).
    pub long_term_feerate: f32,
    /// The minimum absolute fee (in satoshis).
    pub min_absolute_fee: u64,

    /// The weight of the template transaction, inclusive of `nVersion`, `nLockTime`, recipient
    /// `vout`s and the first 1 bytes of `vin_len` and `vout_len`.
    pub base_weight: u32,

    /// The weight of the drain (change) output(s).
    pub drain_weight: u32,
    /// The weight of a `txin` used to spend the drain output(s) later on.
    pub drain_spend_weight: u32,
}

impl CoinSelectorOpt {
    pub fn from_weights(base_weight: u32, drain_weight: u32, drain_spend_weight: u32) -> Self {
        Self {
            recipients_sum: 0,
            // 0.25 per wu i.e. 1 sat per byte
            effective_feerate: 0.25,
            long_term_feerate: 0.25,
            min_absolute_fee: 0,
            base_weight,
            drain_weight,
            drain_spend_weight,
        }
    }

    pub fn fund_outputs(
        txouts: &[TxOut],
        drain_outputs: &[TxOut],
        drain_spend_weight: u32,
    ) -> Self {
        let mut temp_tx = Transaction {
            input: vec![],
            version: 1,
            lock_time: LockTime::ZERO.into(),
            output: txouts.to_vec(),
        };
        let base_weight = temp_tx.weight();

        // this awkward calculation is necessary since TxOut doesn't have \.weight()
        let drain_weight = {
            drain_outputs
                .iter()
                .for_each(|txo| temp_tx.output.push(txo.clone()));
            // tx.output.push(drain_outputs.clone());
            temp_tx.weight() - base_weight
        };

        Self {
            recipients_sum: txouts.iter().map(|txout| txout.value).sum(),
            ..Self::from_weights(base_weight as u32, drain_weight as u32, drain_spend_weight)
        }
    }
}

impl CoinSelector {
    pub fn candidates(&self) -> &[InputCandidate] {
        &self.candidates
    }

    pub fn new(candidates: Vec<InputCandidate>, opts: CoinSelectorOpt) -> Self {
        Self {
            candidates,
            selected: Default::default(),
            opts,
        }
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        self.selected.insert(index);
    }

    pub fn current_weight(&self) -> u32 {
        let witness_header_extra_weight = self
            .selected()
            .find(|(_, wv)| wv.segwit_count > 0)
            .map(|_| 2)
            .unwrap_or(0);
        self.opts.base_weight
            + self
                .selected()
                .map(|(_, wv)| wv.weight + TXIN_BASE_WEIGHT)
                .sum::<u32>()
            + witness_header_extra_weight
    }

    pub fn selected(&self) -> impl Iterator<Item = (usize, InputCandidate)> + '_ {
        self.selected
            .iter()
            .map(|index| (*index, self.candidates.get(*index).unwrap().clone()))
    }

    pub fn unselected(&self) -> Vec<usize> {
        let all_indexes = (0..self.candidates.len()).collect::<BTreeSet<_>>();
        all_indexes.difference(&self.selected).cloned().collect()
    }

    pub fn all_selected(&self) -> bool {
        self.selected.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        for next_unselected in self.unselected() {
            self.select(next_unselected)
        }
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        for next_unselected in self.unselected() {
            self.select(next_unselected);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn current_value(&self) -> u64 {
        self.selected().map(|(_, wv)| wv.value).sum::<u64>()
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let base_weight = self.current_weight();

        if self.current_value() < self.opts.recipients_sum {
            return Err(SelectionFailure::InsufficientFunds {
                selected: self.current_value(),
                needed: self.opts.recipients_sum,
            });
        }

        let inputs_minus_outputs = self.current_value() - self.opts.recipients_sum;

        // check fee rate satisfied
        let feerate_without_drain = inputs_minus_outputs as f32 / base_weight as f32;

        // we simply don't have enough fee to acheieve the feerate
        if feerate_without_drain < self.opts.effective_feerate {
            return Err(SelectionFailure::FeerateTooLow {
                needed: self.opts.effective_feerate,
                had: feerate_without_drain,
            });
        }

        if inputs_minus_outputs < self.opts.min_absolute_fee {
            return Err(SelectionFailure::AbsoluteFeeTooLow {
                needed: self.opts.min_absolute_fee,
                had: inputs_minus_outputs,
            });
        }

        let weight_with_drain = base_weight + self.opts.drain_weight;
        let target_fee_with_drain =
            ((self.opts.effective_feerate * weight_with_drain as f32).ceil() as u64)
                .max(self.opts.min_absolute_fee);
        let target_fee_without_drain = ((self.opts.effective_feerate * base_weight as f32).ceil()
            as u64)
            .max(self.opts.min_absolute_fee);

        let (excess, use_drain) = match inputs_minus_outputs.checked_sub(target_fee_with_drain) {
            Some(excess) => (excess, true),
            None => {
                let implied_output_value = self.current_value() - target_fee_without_drain;
                match implied_output_value.checked_sub(self.opts.recipients_sum) {
                    Some(excess) => (excess, false),
                    None => {
                        return Err(SelectionFailure::InsufficientFunds {
                            selected: self.current_value(),
                            needed: target_fee_without_drain + self.opts.recipients_sum,
                        })
                    }
                }
            }
        };

        let (total_weight, fee) = if use_drain {
            (weight_with_drain, target_fee_with_drain)
        } else {
            (base_weight, target_fee_without_drain)
        };

        Ok(Selection {
            selected: self.selected.clone(),
            excess,
            use_drain,
            total_weight,
            fee,
        })
    }
}

#[derive(Clone, Debug)]
pub enum SelectionFailure {
    InsufficientFunds { selected: u64, needed: u64 },
    FeerateTooLow { needed: f32, had: f32 },
    AbsoluteFeeTooLow { needed: u64, had: u64 },
}

impl core::fmt::Display for SelectionFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectionFailure::InsufficientFunds { selected, needed } => write!(
                f,
                "insufficient coins selected, had {} needed {}",
                selected, needed
            ),
            SelectionFailure::FeerateTooLow { needed, had } => {
                write!(f, "feerate too low, needed {}, had {}", needed, had)
            }
            SelectionFailure::AbsoluteFeeTooLow { needed, had } => {
                write!(f, "absolute fee too low, needed {}, had {}", needed, had)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SelectionFailure {}

#[derive(Clone, Debug)]
pub struct Selection {
    pub selected: BTreeSet<usize>,
    pub excess: u64,
    pub fee: u64,
    pub use_drain: bool,
    pub total_weight: u32,
}

impl Selection {
    pub fn apply_selection<'a, T>(
        &'a self,
        candidates: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        self.selected.iter().map(|i| &candidates[*i])
    }
}
