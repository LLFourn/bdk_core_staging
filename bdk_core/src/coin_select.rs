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

    /// Effective value is actual value of the input minus the fee of inclusing the input.
    pub fn effective_value(&self, opts: &CoinSelectorOpt) -> i64 {
        self.value as i64 - (self.weight as f32 * opts.effective_feerate).ceil() as i64
    }

    /// Calculates the `waste` of including this input.
    pub fn waste(&self, opts: &CoinSelectorOpt) -> i64 {
        (self.weight as f32 * (opts.effective_feerate - opts.long_term_feerate)).ceil() as i64
    }
}

impl core::ops::Add for InputCandidate {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            input_count: self.input_count + rhs.input_count,
            segwit_count: self.segwit_count + rhs.segwit_count,
            value: self.value + rhs.value,
            weight: self.weight + rhs.weight,
        }
    }
}

impl core::ops::Sub for InputCandidate {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            input_count: self.input_count - rhs.input_count,
            segwit_count: self.segwit_count - rhs.segwit_count,
            value: self.value - rhs.value,
            weight: self.weight - rhs.weight,
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

impl core::iter::Sum for InputCandidate {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::empty(), |acc, v| acc + v)
    }
}

#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    opts: &'a CoinSelectorOpt,
    candidates: &'a Vec<InputCandidate>,
    selected: BTreeSet<usize>,
    selected_sum: InputCandidate,
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

    /// The weight introduced when we include the drain (change) output(s).
    /// This should account for the weight difference of the `vout_len` varint.
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

    /// Calculates the "cost of change": cost of creating drain output + cost of spending the drain
    /// output in the future.
    pub fn drain_cost(&self) -> u64 {
        ((self.effective_feerate * self.drain_weight as f32).ceil()
            + (self.long_term_feerate * self.drain_spend_weight as f32).ceil()) as u64
    }

    /// Selection target should be `recipients_sum + base_weight * effective_feerate`
    ///
    /// If `include_segwit == true`, 2 additional WUs are included in `base_weight` to represent the
    /// segwit `marker` and `flag` fields.
    ///
    /// `max_input_count` determines if additional WUs are to be included in `base_weight` due to
    /// `vin_len` varint weight changes.
    pub fn effective_target(&self, include_segwit: bool, max_input_count: usize) -> i64 {
        let base_weight = self.base_weight
            // additional weight from segwit headers
            + if include_segwit { 2_u32 } else { 0_u32 }
            // additional weight from `vin_len` varint changes
            + (varint_size(max_input_count) - 1) * 4;

        self.recipients_sum as i64 + (base_weight as f32 * self.effective_feerate).ceil() as i64
    }
}

impl<'a> CoinSelector<'a> {
    pub fn new(candidates: &'a Vec<InputCandidate>, opts: &'a CoinSelectorOpt) -> Self {
        Self {
            opts,
            candidates,
            selected: Default::default(),
            selected_sum: InputCandidate::empty(),
        }
    }

    pub fn options(&self) -> &CoinSelectorOpt {
        &self.opts
    }

    pub fn candidates(&self) -> &[InputCandidate] {
        &self.candidates
    }

    pub fn candidate(&self, index: usize) -> &InputCandidate {
        &self.candidates[index]
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected.contains(&index)
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected.insert(index) {
            self.selected_sum += self.candidates[index];
        }
    }

    pub fn deselect(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected.remove(&index) {
            self.selected_sum -= self.candidates[index];
        }
    }

    /// Sum of the selected candidate inputs.
    pub fn sum(&self) -> &InputCandidate {
        &self.selected_sum
    }

    /// Excess of the current selection.
    /// The value will be negative if selection does not satisfy effective target.
    pub fn excess(&self, effective_target: i64) -> i64 {
        self.selected_sum.effective_value(self.opts) - effective_target
    }

    /// Total fee payed with no drain output.
    pub fn fee(&self) -> i64 {
        self.selected_sum.value as i64 - self.opts.recipients_sum as i64
    }

    /// Total weight of a transaction formed from the current selection of candidates.
    /// This value assumes no drain output.
    pub fn weight(&self) -> u32 {
        let selected = self.sum();
        let is_segwit = selected.segwit_count > 0;

        self.opts.base_weight
            // weight from selected candidate inputs
            + selected.weight
            // additional weight due to the varint that records input count
            + (varint_size(selected.input_count) - 1) * 4
            // additional weight due to spending of segwit UTXOs; we need to include segwit 
            // `marker` and `flag` fields
            + if is_segwit { 2_u32 } else { 0_u32 }
    }

    pub fn iter_selected_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        self.selected.iter().cloned()
    }

    pub fn iter_unselected_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.candidates.len()).filter(|index| !self.selected.contains(index))
    }

    pub fn iter_selected(&self) -> impl Iterator<Item = &InputCandidate> + '_ {
        self.iter_selected_indexes()
            .map(|index| &self.candidates[index])
    }

    pub fn iter_unselected(&self) -> impl Iterator<Item = &InputCandidate> + '_ {
        self.iter_selected_indexes()
            .map(|index| &self.candidates[index])
    }

    pub fn all_selected(&self) -> bool {
        self.selected.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        let unselected = self.iter_unselected_indexes().collect::<Vec<_>>();
        for index in unselected {
            self.select(index)
        }
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        let unselected = self.iter_unselected_indexes().collect::<Vec<_>>();

        for unselected_index in unselected {
            self.select(unselected_index);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let selected = self.sum();
        let base_weight = self.weight();

        if selected.value < self.opts.recipients_sum {
            return Err(SelectionFailure::InsufficientFunds {
                selected: selected.value,
                needed: self.opts.recipients_sum,
            });
        }

        let inputs_minus_outputs = selected.value - self.opts.recipients_sum;

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

pub fn select_coins_bnb(max_tries: usize, mut selection: CoinSelector) -> Option<CoinSelector> {
    let opts = selection.options().clone();

    let (target_value, mut remaining_value) = {
        let remaining = selection.iter_unselected().cloned().sum::<InputCandidate>();
        let all = *selection.sum() + remaining;

        // we want to avoid undershooting the `target_value` so it is set as the "largest possibe"
        // value given our candidates
        let target_value = opts.effective_target(all.segwit_count > 0, all.input_count);
        let remaining_value = remaining.effective_value(&opts);

        // ensure we have enough to select with
        if remaining_value < target_value {
            return None;
        }

        (target_value, remaining_value)
    };

    let cost_of_change = opts.drain_cost() as i64;
    let feerate_decreasing = opts.effective_feerate > opts.long_term_feerate;
    let upper_bound = target_value + cost_of_change;

    // prepare pool
    let pool = {
        let mut pool = selection.iter_unselected_indexes().collect::<Vec<_>>();
        pool.sort_unstable_by(|&a, &b| {
            let a = selection.candidate(a).effective_value(&opts);
            let b = selection.candidate(b).effective_value(&opts);
            b.cmp(&a)
        });
        pool
    };
    // pool position
    let mut pos = 0_usize;

    // our best solution (least waste, within bounds)
    let mut best_selection = Option::<CoinSelector>::None;

    for try_index in 0..max_tries {
        if try_index > 0 {
            pos += 1;
        }

        let backtrack = {
            let current_value = selection.sum().effective_value(&opts);
            let current_waste = selection.sum().waste(&opts) + selection.excess(target_value);
            let best_waste = best_selection
                .as_ref()
                .map(|b| b.sum().waste(&opts) + b.excess(target_value))
                .unwrap_or(i64::MAX);

            // backtrack if:
            // * value remaining in branch is not enough to reach target value (lower bound)
            // * current selected value surpasses upper bound
            if current_value + remaining_value < target_value || current_value > upper_bound {
                true
            } else if feerate_decreasing && current_waste >= best_waste {
                // with a decreasing feerate, selecting a new input will always decrease the waste
                // however, we want to find a solution with no change output
                true
            } else if current_value >= target_value {
                // we have found a solution, but is it better than our best?
                if current_waste <= best_waste {
                    best_selection.replace(selection.clone());
                }
                true
            } else {
                false
            }
        };

        if backtrack {
            let last_selected_pos = (pos - 1..0).find(|&pos| {
                remaining_value += selection.candidate(pool[pos]).effective_value(&opts);
                selection.is_selected(pool[pos])
            });

            match last_selected_pos {
                Some(last_selected_pos) => {
                    pos = last_selected_pos;
                    selection.deselect(pool[pos]);
                }
                None => break, // nothing is selected, all solutions searched
            }
        } else {
            let candidate = selection.candidate(pool[pos]);
            remaining_value -= candidate.effective_value(&opts);

            // avoid selection if previous position was excluded and has the same value and weight
            if !selection.is_selected(pool[pos - 1]) {
                let prev_candidate = selection.candidate(pool[pos - 1]);
                if candidate.effective_value(&opts) == prev_candidate.effective_value(&opts)
                    && candidate.weight == prev_candidate.weight
                {
                    continue;
                }
            }

            // select
            selection.select(pool[pos]);
        }
    }

    best_selection
}
