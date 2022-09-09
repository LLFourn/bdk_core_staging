use crate::{collections::BTreeSet, Vec};
use bitcoin::{LockTime, Transaction, TxOut};

pub const TXIN_FIXED_WEIGHT: u32 = (32 + 4 + 4) * 4;

#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    opts: &'a CoinSelectorOpt,
    candidates: &'a Vec<InputCandidate>,

    /* The following fields record the selection state */
    selected_indexes: BTreeSet<usize>, // indexes of selected input candidates
    selected_state: InputCandidate,    // state of the selected inputs
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputCandidate {
    /// Number of inputs contained within this [`InputCandidate`].
    /// If we are using single UTXOs as candidates, this would be 1.
    /// If we are working in `OutputGroup`s (as done in Bitcoin Core), this would be > 1.
    pub input_count: usize,
    /// Whether at least one input of this [`InputCandidate`] is spending a segwit output.
    pub segwit_count: usize,
    /// Total value of these input(s).
    pub value: u64,
    /// Weight of these input(s): `prevout + nSequence + scriptSig + scriptWitness` per input.
    pub weight: u32,
}

impl InputCandidate {
    /// New [`InputCandidate`] where `effective_value` is calculated from fee defined in `opts`.
    pub fn new(input_count: usize, value: u64, weight: u32, is_segwit: bool) -> Self {
        assert!(
            input_count > 0,
            "InputCandidate does not make sense with 0 inputs"
        );
        Self {
            input_count,
            value,
            weight,
            segwit_count: if is_segwit { 1 } else { 0 },
        }
    }

    pub fn empty() -> Self {
        Self {
            input_count: 0,
            segwit_count: 0,
            value: 0,
            weight: 0,
        }
    }

    pub fn effective_value(&self, opts: &CoinSelectorOpt) -> i64 {
        self.value as i64 - (self.weight as f32 * opts.effective_feerate).ceil() as i64
    }

    /// Calculates the `waste` of including this input.
    pub fn waste(&self, opts: &CoinSelectorOpt) -> i64 {
        (self.weight as f32 * (opts.effective_feerate - opts.long_term_feerate)).ceil() as i64
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

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// The sum of recipient output values (in satoshis).
    pub recipients_sum: u64,

    /// The feerate we should try and achieve in sats per weight unit.
    pub effective_feerate: f32,
    /// The long term feerate if we are to spend an input in the future instead (in sats/wu).
    /// This is used for calculating waste.
    pub long_term_feerate: f32,
    /// The minimum absolute fee (in satoshis).
    pub min_absolute_fee: u64,

    /// Additional weight if we use the drain (change) output(s).
    pub drain_weight: u32,
    /// Weight of a `txin` used to spend the drain output(s) later on.
    pub drain_spend_weight: u32,

    /// The fixed weight of the template transaction, inclusive of: `nVersion`, `nLockTime`
    /// fixed `vout`s and the first bytes of `vin_len` and `vout_len`.
    ///
    /// Weight of the drain output is not included.
    pub fixed_weight: u32,
}

impl CoinSelectorOpt {
    pub fn from_weights(fixed_weight: u32, drain_weight: u32, drain_spend_weight: u32) -> Self {
        Self {
            recipients_sum: 0,
            // 0.25 per wu i.e. 1 sat per byte
            effective_feerate: 0.25,
            long_term_feerate: 0.25,
            min_absolute_fee: 0,
            drain_weight,
            drain_spend_weight,
            fixed_weight,
        }
    }

    pub fn fund_outputs(
        txouts: &[TxOut],
        drain_outputs: &[TxOut],
        drain_spend_weight: u32,
    ) -> Self {
        let mut tx = Transaction {
            input: vec![],
            version: 1,
            lock_time: LockTime::ZERO.into(),
            output: txouts.to_vec(),
        };
        let fixed_weight = tx.weight();

        // this awkward calculation is necessary since TxOut doesn't have \.weight()
        let drain_weight = {
            drain_outputs
                .iter()
                .for_each(|txo| tx.output.push(txo.clone()));
            tx.weight() - fixed_weight
        };

        Self {
            recipients_sum: txouts.iter().map(|txout| txout.value).sum(),
            ..Self::from_weights(fixed_weight as u32, drain_weight as u32, drain_spend_weight)
        }
    }

    /// Calculates the "cost of change": cost of creating drain output + cost of spending the drain
    /// output in the future.
    pub fn drain_cost(&self) -> u64 {
        ((self.effective_feerate * self.drain_weight as f32).ceil()
            + (self.long_term_feerate * self.drain_spend_weight as f32).ceil()) as u64
    }

    /// This is the extra weight of the `txin_count` variable (which is a `varint`), when we
    /// introduce inputs on top of the "fixed" input count.
    pub fn extra_varint_weight(&self, total_input_count: usize) -> u32 {
        (varint_size(total_input_count) - 1) * 4
    }

    /// Selection target should be `recipients_sum + fixed_weight * effective_feerate`
    pub fn target_effective_value(&self) -> i64 {
        self.recipients_sum as i64
            + (self.fixed_weight as f32 * self.effective_feerate).ceil() as i64
    }
}

impl<'a> CoinSelector<'a> {
    pub fn new(candidates: &'a Vec<InputCandidate>, opts: &'a CoinSelectorOpt) -> Self {
        Self {
            opts,
            candidates,

            selected_indexes: Default::default(),
            selected_state: InputCandidate::empty(),
        }
    }

    pub fn candidates(&self) -> &[InputCandidate] {
        &self.candidates
    }

    pub fn candidate(&self, index: usize) -> &InputCandidate {
        assert!(index < self.candidates.len());
        &self.candidates[index]
    }

    pub fn options(&self) -> &CoinSelectorOpt {
        self.opts
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected_indexes.insert(index) {
            self.selected_state += self.candidates[index];
        }
    }

    pub fn deselect(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected_indexes.remove(&index) {
            self.selected_state -= self.candidates[index];
        }
    }

    /// Returns the current state of all inputs in the current selection.
    pub fn state(&self) -> &InputCandidate {
        &self.selected_state
    }

    pub fn excess(&self) -> i64 {
        self.selected_state.effective_value(self.opts) - self.opts.target_effective_value()
    }

    pub fn current_weight_without_drain(&self) -> u32 {
        let inputs = self.state();
        let is_segwit = inputs.segwit_count > 0;

        let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
        let extra_varint_weight = self.opts.extra_varint_weight(inputs.input_count);

        self.opts.fixed_weight + inputs.weight + extra_witness_weight + extra_varint_weight
    }

    pub fn iter_selected(&self) -> impl Iterator<Item = (usize, InputCandidate)> + '_ {
        self.selected_indexes
            .iter()
            .map(|&index| (index, self.candidates[index]))
    }

    pub fn iter_unselected(&self) -> impl Iterator<Item = (usize, InputCandidate)> + '_ {
        self.candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.selected_indexes.contains(index))
            .map(|(index, c)| (index, *c))
    }

    pub fn all_selected(&self) -> bool {
        self.selected_indexes.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        let unselected = self.iter_unselected().map(|(i, _)| i).collect::<Vec<_>>();
        for index in unselected {
            self.select(index);
        }
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        let unselected = self.iter_unselected().map(|(i, _)| i).collect::<Vec<_>>();
        for index in unselected {
            self.select(index);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let selected = self.state();

        // this is the tx weight without drain.
        let base_weight = {
            let is_segwit = selected.segwit_count > 0;
            let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
            let extra_varint_weight = self.opts.extra_varint_weight(selected.input_count);
            self.opts.fixed_weight + selected.weight + extra_witness_weight + extra_varint_weight
        };

        if selected.value < self.opts.recipients_sum {
            return Err(SelectionFailure::InsufficientFunds {
                selected: selected.value,
                needed: self.opts.recipients_sum,
            });
        }

        let inputs_minus_outputs = selected.value - self.opts.recipients_sum;

        // check fee rate satisfied
        let feerate_without_drain = inputs_minus_outputs as f32 / base_weight as f32;

        // we simply don't have enough fee to achieve the feerate
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
                let implied_output_value = selected.value - target_fee_without_drain;
                match implied_output_value.checked_sub(self.opts.recipients_sum) {
                    Some(excess) => (excess, false),
                    None => {
                        return Err(SelectionFailure::InsufficientFunds {
                            selected: selected.value,
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

        // `waste` is the waste of spending the inputs now (with the current selection), as opposed
        // to spending it later.
        let waste = selected.waste(self.opts)
            + if use_drain {
                self.opts.drain_cost()
            } else {
                excess
            } as i64;

        Ok(Selection {
            selected: self.selected_indexes.clone(),
            excess,
            use_drain,
            total_weight,
            fee,
            waste,
        })
    }
}

#[derive(Clone, Debug)]
pub enum SelectionFailure {
    InsufficientFunds { selected: u64, needed: u64 },
    FeerateTooLow { needed: f32, had: f32 },
    AbsoluteFeeTooLow { needed: u64, had: u64 },
    NoSolution,
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
            Self::NoSolution => {
                write!(f, "algorithm cannot find a solution")
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
    pub waste: i64,
}

impl Selection {
    pub fn apply_selection<'a, T>(
        &'a self,
        candidates: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        self.selected.iter().map(|i| &candidates[*i])
    }
}

/* HELPERS */

fn varint_size(v: usize) -> u32 {
    if v <= 0xfc {
        return 1;
    }
    if v <= 0xffff {
        return 3;
    }
    if v <= 0xffff_ffff {
        return 5;
    }
    return 9;
}

/* ALGORITHMS */

pub fn select_coins_bnb(current: &mut CoinSelector) -> Result<Selection, SelectionFailure> {
    const MAX_MONEY: i64 = 2_100_000_000_000_000;
    const MAX_TRIES: usize = 100_000;

    let target_value = current.options().target_effective_value();
    let cost_of_change = current.options().drain_cost() as i64;

    // remaining value of the current branch
    let mut remaining_value = current
        .iter_unselected()
        .map(|(_, ic)| ic.effective_value(current.options()))
        .sum::<i64>();
    // ensure we have enough to select with
    if remaining_value < target_value {
        todo!("properly handle this error")
    }

    // our best solution (start with the worst possible solution)
    let mut best = Option::<CoinSelector>::None;

    // sort unselected index pool in descending order in terms of effective value
    let mut pool_index = 0_usize;
    let mut selected_pool_indexes = Vec::<usize>::with_capacity(current.candidates().len());
    let pool = {
        let mut pool = current.iter_unselected().collect::<Vec<_>>();
        pool.sort_by(|(_, ca), (_, cb)| {
            cb.effective_value(current.options())
                .cmp(&ca.effective_value(current.options()))
        });
        pool
    };

    // depth-first loop
    for try_index in 0..MAX_TRIES {
        if try_index > 0 {
            pool_index += 1;
        }

        // conditions for starting a backtrack
        let backtrack = {
            let opts = current.options();

            let feerate_decreasing = opts.effective_feerate > opts.long_term_feerate;

            let current_value = current.state().effective_value(opts);
            let current_waste = current.state().waste(opts) + current.excess();
            let best_waste = best
                .as_ref()
                .map(|b| b.state().waste(opts) + b.excess())
                .unwrap_or(MAX_MONEY);

            // TODO: Add comments
            if current_value + remaining_value < target_value
                || current_value > target_value + cost_of_change
                || (current.state().waste(opts) > best_waste && feerate_decreasing)
            {
                true
            } else if current_value >= target_value {
                if current_waste <= best_waste {
                    best.replace(current.clone());
                }

                true
            } else {
                false
            }
        };

        if backtrack {
            if current.state().value == 0 {
                break;
            }

            pool_index -= 1;
            while pool_index > *selected_pool_indexes.last().unwrap() {
                remaining_value += pool[pool_index].1.effective_value(current.options());
                pool_index -= 1;
            }
            assert_eq!(pool_index, *selected_pool_indexes.last().unwrap());

            let candidate_index = pool[pool_index].0;
            current.deselect(candidate_index);
            selected_pool_indexes.pop();

            continue;
        }

        let (candidate_index, candidate) = pool[pool_index];
        remaining_value -= candidate.effective_value(current.options());

        if current.state().value == 0
            || pool_index - 1 == *selected_pool_indexes.last().unwrap()
            || candidate.effective_value(current.options())
                != pool[pool_index - 1].1.effective_value(current.options())
            || candidate.weight != pool[pool_index - 1].1.weight
        {
            current.select(candidate_index);
            selected_pool_indexes.push(pool_index);
        }
    }

    let selection = best
        .as_ref()
        .ok_or(SelectionFailure::NoSolution)?
        .finish()?;

    assert_eq!(
        selection.waste,
        {
            let best = best.as_ref().unwrap();
            best.state().waste(current.options()) + best.excess()
        },
        "waste does not match up"
    );

    Ok(selection)
}
