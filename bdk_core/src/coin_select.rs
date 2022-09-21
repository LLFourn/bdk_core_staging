use crate::{
    collections::{BTreeSet, HashMap},
    Vec,
};
use bitcoin::{LockTime, Transaction, TxOut};

/// Txin "base" fields include `outpoint` (32+4) and `nSequence` (4). This does not include
/// `scriptSigLen` or `scriptSig`.
pub const TXIN_BASE_WEIGHT: u32 = (32 + 4 + 4) * 4;

/// [`CoinSelector`] is responsible for selecting and deselecting from a set of canididates.
#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    opts: &'a CoinSelectorOpt,
    candidates: Vec<(usize, &'a WeightedValue)>,
    selected_pos: BTreeSet<usize>,
}

/// A [`WeightedValue`] represents an input candidate for [`CoinSelector`]. This can either be a
/// single UTXO, or a group of UTXOs that should be spent together.
#[derive(Debug, Clone, Copy)]
pub struct WeightedValue {
    /// Total value of the UTXO(s) that this [`WeightedValue`] represents.
    pub value: u64,
    /// Total weight of including this/these UTXO(s).
    /// `txin` fields: `prevout`, `nSequence`, `scriptSigLen`, `scriptSig`, `scriptWitnessLen`,
    /// `scriptWitness` should all be included.
    pub weight: u32,
    /// Total number of inputs; so we can calculate extra `varint` weight due to `vin` len changes.
    pub input_count: usize,
    /// Whether this [`WeightedValue`] contains at least one segwit spend.
    pub is_segwit: bool,
}

impl WeightedValue {
    /// Create a new [`WeightedValue`] that represents a single input.
    ///
    /// `satisfaction_weight` is the weight of `scriptSigLen + scriptSig + scriptWitnessLen +
    /// scriptWitness`.
    pub fn new(value: u64, satisfaction_weight: u32, is_segwit: bool) -> WeightedValue {
        let weight = TXIN_BASE_WEIGHT + satisfaction_weight;
        WeightedValue {
            value,
            weight,
            input_count: 1,
            is_segwit,
        }
    }

    /// Effective feerate of this input candidate.
    /// `actual_value - input_weight * feerate`
    pub fn effective_value(&self, opts: &CoinSelectorOpt) -> i64 {
        // we prefer undershooting the candidate's effective value
        self.value as i64 - (self.weight as f32 * opts.target_feerate).ceil() as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// The value we need to select.
    pub target_value: u64,
    /// Additional leeway for the target value.
    pub max_extra_target: u64,

    /// The feerate we should try and achieve in sats per weight unit.
    pub target_feerate: f32,
    /// The feerate
    pub long_term_feerate: Option<f32>,
    /// The minimum absolute fee.
    pub min_absolute_fee: u64,

    /// The weight of the template transaction including fixed inputs and outputs.
    pub base_weight: u32,
    /// Additional weight if we include the drain (change) output.
    pub drain_weight: u32,
    /// TODO
    pub spend_drain_weight: u32,

    /// Minimum value allowed for a drain (change) output.
    pub min_drain_value: u64,
}

impl CoinSelectorOpt {
    fn from_weights(base_weight: u32, drain_weight: u32, spend_drain_weight: u32) -> Self {
        // 0.25 sats/wu == 1 sat/vb
        let target_feerate = 0.25_f32;

        // set `min_drain_value` to dust limit
        let min_drain_value =
            3 * ((drain_weight + spend_drain_weight) as f32 * target_feerate) as u64;

        Self {
            target_value: 0,
            max_extra_target: 0,
            target_feerate,
            long_term_feerate: None,
            min_absolute_fee: 0,
            base_weight,
            drain_weight,
            spend_drain_weight,
            min_drain_value,
        }
    }

    pub fn fund_outputs(
        txouts: &[TxOut],
        drain_output: &TxOut,
        drain_satisfaction_weight: u32,
    ) -> Self {
        let mut tx = Transaction {
            input: vec![],
            version: 1,
            lock_time: LockTime::ZERO.into(),
            output: txouts.to_vec(),
        };
        let base_weight = tx.weight();
        // this awkward calculation is necessary since TxOut doesn't have \.weight()
        let drain_weight = {
            tx.output.push(drain_output.clone());
            tx.weight() - base_weight
        };
        Self {
            target_value: txouts.iter().map(|txout| txout.value).sum(),
            ..Self::from_weights(
                base_weight as u32,
                drain_weight as u32,
                TXIN_BASE_WEIGHT + drain_satisfaction_weight,
            )
        }
    }

    pub fn long_term_feerate(&self) -> f32 {
        self.long_term_feerate.unwrap_or(self.target_feerate)
    }

    pub fn drain_waste(&self) -> i64 {
        (self.drain_weight as f32 * self.target_feerate
            + self.spend_drain_weight as f32 * self.long_term_feerate()) as i64
    }
}

impl<'a> CoinSelector<'a> {
    pub fn new(candidates: &'a Vec<WeightedValue>, opts: &'a CoinSelectorOpt) -> Self {
        Self {
            candidates: candidates.iter().enumerate().collect(),
            selected_pos: Default::default(),
            opts,
        }
    }

    pub fn opts(&self) -> &CoinSelectorOpt {
        &self.opts
    }

    pub fn candidate_at(&self, pos: usize) -> &WeightedValue {
        self.candidates[pos].1
    }

    pub fn sort_candidates<F>(&mut self, compare: F)
    where
        F: FnMut(&(usize, &WeightedValue), &(usize, &WeightedValue)) -> core::cmp::Ordering,
    {
        self.candidates.sort_unstable_by(compare)
    }

    pub fn iter_candidates(&self) -> impl Iterator<Item = &WeightedValue> + '_ {
        self.candidates.iter().map(|(_, c)| *c)
    }

    pub fn iter_selected_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.selected_pos.iter().cloned()
    }

    pub fn iter_selected(&self) -> impl Iterator<Item = &WeightedValue> + '_ {
        self.selected_pos.iter().map(|&pos| self.candidates[pos].1)
    }

    pub fn iter_unselected_positions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.candidates.len()).filter(|pos| !self.selected_pos.contains(pos))
    }

    pub fn iter_unselected(&self) -> impl Iterator<Item = &WeightedValue> + '_ {
        self.iter_unselected_positions()
            .map(|pos| self.candidates[pos].1)
    }

    pub fn is_selected(&self, pos: usize) -> bool {
        self.selected_pos.contains(&pos)
    }

    pub fn is_none_selected(&self) -> bool {
        self.selected_pos.is_empty()
    }

    pub fn is_all_selected(&self) -> bool {
        self.selected_pos.len() == self.candidates.len()
    }

    /// Weight sum of all selected inputs.
    pub fn selected_weight(&self) -> u32 {
        self.selected_pos
            .iter()
            .map(|&pos| self.candidates[pos].1.weight)
            .sum()
    }

    /// Effective value sum of all selected inputs.
    pub fn selected_effective_value(&self) -> i64 {
        self.selected_pos
            .iter()
            .map(|&pos| self.candidates[pos].1.effective_value(&self.opts))
            .sum()
    }

    /// Absolute value sum of all selected inputs.
    pub fn selected_absolute_value(&self) -> u64 {
        self.selected_pos
            .iter()
            .map(|&pos| self.candidates[pos].1.value)
            .sum()
    }

    /// Waste sum of all selected inputs.
    pub fn selected_waste(&self) -> i64 {
        (self.selected_weight() as f32 * (self.opts.target_feerate - self.opts.long_term_feerate()))
            as i64
    }

    /// Current weight of template tx + selected inputs.
    pub fn current_weight(&self) -> u32 {
        let witness_header_extra_weight = self
            .iter_selected()
            .find(|wv| wv.is_segwit)
            .map(|_| 2)
            .unwrap_or(0);
        let vin_count_varint_extra_weight = {
            let input_count = self.iter_selected().map(|wv| wv.input_count).sum::<usize>();
            (varint_size(input_count) - 1) * 4
        };
        self.opts.base_weight
            + self.selected_weight()
            + witness_header_extra_weight
            + vin_count_varint_extra_weight
    }

    /// Current excess.
    pub fn current_excess(&self) -> i64 {
        let effective_target = self.opts.target_value as i64
            + (self.opts.base_weight as f32 * self.opts.target_feerate) as i64;
        self.selected_effective_value() - effective_target
    }

    pub fn select(&mut self, pos: usize) {
        assert!(pos < self.candidates.len());
        self.selected_pos.insert(pos);
    }

    pub fn deselect(&mut self, pos: usize) {
        self.selected_pos.remove(&pos);
    }

    pub fn select_all(&mut self) {
        self.selected_pos = (0..self.candidates.len()).collect();
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        let unselected_pos = self.iter_unselected_positions().collect::<Vec<_>>();

        for pos in unselected_pos {
            self.select(pos);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let weight_without_drain = self.current_weight();
        let weight_with_drain = weight_without_drain + self.opts.drain_weight;

        let fee_without_drain =
            (weight_without_drain as f32 * self.opts.target_feerate).ceil() as u64;
        let fee_with_drain = (weight_with_drain as f32 * self.opts.target_feerate).ceil() as u64;

        let inputs_minus_outputs = {
            let target_value = self.opts.target_value;
            let selected = self.selected_absolute_value();

            // find the largest unsatisfied constraint (if any), and return error of that constraint
            [
                (
                    SelectionConstraint::TargetValue,
                    target_value.saturating_sub(selected),
                ),
                (
                    SelectionConstraint::TargetFee,
                    (target_value + fee_without_drain).saturating_sub(selected),
                ),
                (
                    SelectionConstraint::MinAbsoluteFee,
                    (target_value + self.opts.min_absolute_fee).saturating_sub(selected),
                ),
            ]
            .into_iter()
            .filter(|&(_, v)| v > 0)
            .max_by_key(|&(_, v)| v)
            .map_or(Ok(()), |(constraint, missing)| {
                Err(SelectionFailure::InsufficientFunds {
                    selected,
                    missing,
                    constraint,
                })
            })?;

            (selected - target_value) as u64
        };

        let fee_without_drain = fee_without_drain.max(self.opts.min_absolute_fee);
        let fee_with_drain = fee_with_drain.max(self.opts.min_absolute_fee);

        let excess_without_drain = inputs_minus_outputs - fee_without_drain;
        let input_waste = self.selected_waste();

        // begin preparing excess strategies for final selection
        let mut excess_strategies = HashMap::new();

        // no drain, excess to fee
        excess_strategies.insert(
            ExcessStrategyKind::ToFee,
            ExcessStrategy {
                recipient_value: self.opts.target_value,
                drain_value: None,
                fee: fee_without_drain + excess_without_drain,
                weight: weight_without_drain,
                waste: input_waste + excess_without_drain as i64,
            },
        );

        // no drain, excess to recipient
        // if `excess == 0`, this result will be the same as the previous, so we don't consider it
        // if `max_extra_target == 0`, there is no leeway for this strategy
        if excess_without_drain > 0 && self.opts.max_extra_target > 0 {
            let extra_recipient_value =
                core::cmp::min(self.opts.max_extra_target, excess_without_drain);
            let extra_fee = excess_without_drain - extra_recipient_value;
            excess_strategies.insert(
                ExcessStrategyKind::ToRecipient,
                ExcessStrategy {
                    recipient_value: self.opts.target_value + extra_recipient_value,
                    drain_value: None,
                    fee: fee_without_drain + extra_fee,
                    weight: weight_without_drain,
                    waste: input_waste + extra_fee as i64,
                },
            );
        }

        // with drain
        if inputs_minus_outputs >= fee_with_drain + self.opts.min_drain_value {
            excess_strategies.insert(
                ExcessStrategyKind::ToDrain,
                ExcessStrategy {
                    recipient_value: self.opts.target_value,
                    drain_value: Some(inputs_minus_outputs.saturating_sub(fee_with_drain)),
                    fee: fee_with_drain,
                    weight: weight_with_drain,
                    waste: input_waste + self.opts.drain_waste(),
                },
            );
        }

        Ok(Selection {
            selected: self
                .selected_pos
                .iter()
                .map(|&pos| self.candidates[pos].0)
                .collect(),
            excess: excess_without_drain,
            excess_strategies,
        })
    }
}

#[derive(Clone, Debug)]
pub enum SelectionFailure {
    InsufficientFunds {
        selected: u64,
        missing: u64,
        constraint: SelectionConstraint,
    },
}

impl core::fmt::Display for SelectionFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectionFailure::InsufficientFunds {
                selected,
                missing,
                constraint,
            } => write!(
                f,
                "insufficient coins selected; selected={}, missing={}, unsatisfied_constraint={:?}",
                selected, missing, constraint
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SelectionFailure {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectionConstraint {
    /// The target is not met
    TargetValue,
    /// The target fee (given the feerate) is not met
    TargetFee,
    /// Min absolute fee in not met
    MinAbsoluteFee,
}

impl core::fmt::Display for SelectionConstraint {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectionConstraint::TargetValue => core::write!(f, "target_value"),
            SelectionConstraint::TargetFee => core::write!(f, "target_fee"),
            SelectionConstraint::MinAbsoluteFee => core::write!(f, "min_absolute_fee"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub selected: BTreeSet<usize>,
    pub excess: u64,
    pub excess_strategies: HashMap<ExcessStrategyKind, ExcessStrategy>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, core::hash::Hash)]
pub enum ExcessStrategyKind {
    ToFee,
    ToRecipient,
    ToDrain,
}

#[derive(Clone, Copy, Debug)]
pub struct ExcessStrategy {
    pub recipient_value: u64,
    pub drain_value: Option<u64>,
    pub fee: u64,
    pub weight: u32,
    pub waste: i64,
}

impl core::fmt::Display for ExcessStrategyKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExcessStrategyKind::ToFee => core::write!(f, "to_fee"),
            ExcessStrategyKind::ToRecipient => core::write!(f, "to_recipient"),
            ExcessStrategyKind::ToDrain => core::write!(f, "to_drain"),
        }
    }
}

impl ExcessStrategy {
    /// Returns feerate in sats/wu.
    pub fn feerate(&self) -> f32 {
        self.fee as f32 / self.weight as f32
    }
}

impl Selection {
    pub fn apply_selection<'a, T>(
        &'a self,
        candidates: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        self.selected.iter().map(|i| &candidates[*i])
    }

    /// Returns the [`ExcessStrategy`] that results in the least waste.
    pub fn best_strategy(&self) -> (&ExcessStrategyKind, &ExcessStrategy) {
        self.excess_strategies
            .iter()
            .min_by_key(|&(_, a)| a.waste)
            .expect("selection has no excess strategy")
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

/// This is a variation of the Branch and Bound Coin Selection algorithm designed by Murch (as seen
/// in Bitcoin Core).
///
/// The differences are as follows:
/// * In additional to working with effective values, we also work with absolute values.
///   This way, we can use bounds of absolute values to enforce `min_absolute_fee` (which is used by
///   RBF), and `max_extra_target` (which can be used to increase the possible solution set, given
///   that the sender is okay with sending extra to the receiver).
///
/// Murch's Master Thesis: https://murch.one/wp-content/uploads/2016/11/erhardt2016coinselection.pdf
/// Bitcoin Core Implementation: https://github.com/bitcoin/bitcoin/blob/23.x/src/wallet/coinselection.cpp#L65
pub fn coin_select_bnb(max_tries: usize, selection: &mut CoinSelector) -> bool {
    let opts = selection.opts().clone();

    let base_weight = {
        let (has_segwit, max_input_count) = selection
            .iter_candidates()
            .fold((false, 0_usize), |(is_segwit, input_count), c| {
                (is_segwit || c.is_segwit, input_count + c.input_count)
            });

        opts.base_weight
            + if has_segwit { 2_u32 } else { 0_u32 }
            + (varint_size(max_input_count) - 1) * 4
    };

    let pool = {
        // TODO: Another optimisation we could do is figure out candidate with smallest waste, and
        // if we find a result with waste equal to this, we can just break.
        let mut pool = selection
            .iter_unselected_positions()
            .filter(|&index| selection.candidate_at(index).effective_value(&opts) > 0)
            .collect::<Vec<_>>();
        // sort by descending effective value
        pool.sort_unstable_by(|&a, &b| {
            let a = selection.candidate_at(a).effective_value(&opts);
            let b = selection.candidate_at(b).effective_value(&opts);
            b.cmp(&a)
        });
        pool
    };

    let mut pos = 0_usize;

    let target_value =
        opts.target_value as i64 + (base_weight as f32 * opts.target_feerate).ceil() as i64;
    let mut remaining_value = pool
        .iter()
        .map(|&index| selection.candidate_at(index).effective_value(&opts))
        .sum::<i64>();

    if selection.selected_effective_value() + remaining_value < target_value {
        return false;
    }

    let abs_target_value = opts.target_value + opts.min_absolute_fee;
    let mut abs_remaining_value = pool
        .iter()
        .map(|&i| selection.candidate_at(i).value)
        .sum::<u64>();

    if selection.selected_absolute_value() + abs_remaining_value < abs_target_value {
        return false;
    }

    let feerate_decreasing = opts.target_feerate > opts.long_term_feerate();

    let upper_bound = target_value + opts.drain_waste();
    let abs_upper_bound =
        abs_target_value + (opts.drain_weight as f32 * opts.target_feerate) as u64;

    // the solution (if any) with the least `waste` is stored here
    let mut best_selection = Option::<CoinSelector>::None;

    for try_index in 0..max_tries {
        // increment `pos`, but only after the first round
        pos += (try_index > 0) as usize;

        let current_value = selection.selected_effective_value();
        let abs_current_value = selection.selected_absolute_value();
        let current_input_waste = selection.selected_waste();

        let best_waste = best_selection
            .as_ref()
            .map(|b| b.selected_waste() + b.current_excess())
            .unwrap_or(i64::MAX);

        // `max_extra_target` is only used as a back-up
        // we only use it for the absolute upper bound when we have no solution
        // but if `max_extra_target` does not surpass `drain_fee`, it can be ignored
        let abs_upper_bound = if best_selection.is_none() {
            core::cmp::max(abs_target_value + opts.max_extra_target, abs_upper_bound)
        } else {
            abs_upper_bound
        };

        // determine if a backtrack is needed for this round...
        let backtrack = if current_value + remaining_value < target_value
            || abs_current_value + abs_remaining_value < abs_target_value
        {
            // remaining value is not enough to reach target
            true
        } else if current_value > upper_bound && abs_current_value > abs_upper_bound {
            // absolute value AND current value both surpasses upper bounds
            true
        } else if feerate_decreasing && current_input_waste > best_waste {
            // when feerate decreases, waste is guaranteed to increase with each new selection,
            // so we should backtrack when we have already surpassed best waste
            true
        } else if current_value >= target_value && abs_current_value >= abs_target_value {
            // we have found a solution, but is it better than our best?
            let current_waste = current_input_waste + current_value - target_value;
            if current_waste <= best_waste {
                // #[cfg(feature = "std")]
                // println!("solution @ try {} with waste {}", try_index, current_waste);
                best_selection.replace(selection.clone());
            }
            true
        } else {
            // no backtrack
            false
        };

        if backtrack {
            // println!("backtrack @ try {}", try_index);
            let last_selected_pos = (0..pos).rev().find(|&pos| {
                let is_selected = selection.is_selected(pool[pos]);
                if !is_selected {
                    let candidate = &selection.candidate_at(pool[pos]);
                    remaining_value += candidate.effective_value(&opts);
                    abs_remaining_value += candidate.value;
                }
                is_selected
            });

            match last_selected_pos {
                Some(last_selected_pos) => {
                    pos = last_selected_pos;
                    selection.deselect(pool[pos]);
                    continue;
                }
                None => break, // nothing is selected, all solutions searched
            }
        }

        let candidate = selection.candidate_at(pool[pos]);
        remaining_value -= candidate.effective_value(&opts);
        abs_remaining_value -= candidate.value;

        // early bailout optimisation:
        // if the candidate at the previous position is NOT selected and has the same weight and
        // value as the current candidate, we skip the current candidate
        if pos > 0 && !selection.is_none_selected() {
            let prev_candidate = selection.candidate_at(pool[pos - 1]);
            if !selection.is_selected(pool[pos - 1])
                && candidate.value == prev_candidate.value
                && candidate.weight == prev_candidate.weight
            {
                continue;
            }
        }

        selection.select(pool[pos]);
    }

    match best_selection {
        Some(best_selection) => {
            *selection = best_selection;
            true
        }
        None => false,
    }
}

#[cfg(feature = "std")]
pub mod evaluate_cs {
    use super::{CoinSelector, ExcessStrategyKind, Selection, Vec};

    pub fn evaluate<F>(
        initial_selector: CoinSelector,
        mut select: F,
    ) -> Result<Evaluation, EvaluationFailure>
    where
        F: FnMut(&mut CoinSelector) -> bool,
    {
        let mut selector = initial_selector.clone();
        let start_time = std::time::SystemTime::now();
        let has_solution = select(&mut selector);
        let elapsed = start_time.elapsed().expect("system time error");

        if has_solution {
            let solution = selector.finish().expect("failed to finish what we started");

            let elapsed_per_candidate = elapsed / selector.candidates.len() as _;

            let waste_vec = solution
                .excess_strategies
                .iter()
                .map(|(_, s)| s.waste)
                .collect::<Vec<_>>();

            let waste_mean = waste_vec.iter().sum::<i64>() as f32 / waste_vec.len() as f32;
            let waste_median = if waste_vec.len() % 2 != 0 {
                waste_vec[waste_vec.len() / 2] as f32
            } else {
                (waste_vec[(waste_vec.len() - 1) / 2] + waste_vec[waste_vec.len() / 2]) as f32 / 2.0
            };

            Ok(Evaluation {
                initial_selector,
                solution,
                elapsed,
                elapsed_per_candidate,
                waste_median,
                waste_mean,
            })
        } else {
            Err(EvaluationFailure {
                initial: initial_selector,
                elapsed,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct Evaluation<'a> {
        pub initial_selector: CoinSelector<'a>,
        pub solution: Selection,

        pub elapsed: std::time::Duration,
        pub elapsed_per_candidate: std::time::Duration,

        pub waste_median: f32,
        pub waste_mean: f32,
    }

    impl<'a> Evaluation<'a> {
        pub fn waste(&self, strategy_kind: ExcessStrategyKind) -> i64 {
            self.solution.excess_strategies[&strategy_kind].waste
        }

        pub fn feerate_offset(&self, strategy_kind: ExcessStrategyKind) -> f32 {
            let target_rate = self.initial_selector.opts.target_feerate;
            let actual_rate = self.solution.excess_strategies[&strategy_kind].feerate();
            actual_rate - target_rate
        }
    }

    impl<'a> core::fmt::Display for Evaluation<'a> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            writeln!(f, "Evaluation:")?;
            writeln!(
                f,
                "\t* Candidates: {}",
                self.initial_selector.candidates.len()
            )?;
            writeln!(
                f,
                "\t* Initial selection: {}",
                self.initial_selector.selected_pos.len()
            )?;
            writeln!(f, "\t* Final selection: {}", self.solution.selected.len())?;
            writeln!(f, "\t* Elapsed: {:?}", self.elapsed)?;
            writeln!(
                f,
                "\t* Elapsed per candidate: {:?}",
                self.elapsed_per_candidate
            )?;
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    pub struct EvaluationFailure<'a> {
        initial: CoinSelector<'a>,
        elapsed: std::time::Duration,
    }

    impl<'a> core::fmt::Display for EvaluationFailure<'a> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                f,
                "cs algorithm failed to find a solution: elapsed={}s target_feerate={}sats/wu",
                self.elapsed.as_secs(),
                self.initial.opts.target_feerate
            )
        }
    }

    impl<'a> std::error::Error for EvaluationFailure<'a> {}
}

#[cfg(test)]
pub mod tester {
    use super::*;
    use bitcoin::{
        secp256k1::{All, Secp256k1},
        TxOut,
    };
    use miniscript::{
        plan::{Assets, Plan},
        Descriptor, DescriptorPublicKey,
    };

    #[derive(Debug, Clone)]
    pub struct TestCandidate {
        pub txo: TxOut,
        pub plan: Plan<DescriptorPublicKey>,
    }

    impl From<TestCandidate> for WeightedValue {
        fn from(test_candidate: TestCandidate) -> Self {
            Self {
                value: test_candidate.txo.value,
                weight: TXIN_BASE_WEIGHT + test_candidate.plan.expected_weight() as u32,
                input_count: 1,
                is_segwit: test_candidate.plan.witness_version().is_some(),
            }
        }
    }

    pub struct Tester {
        descriptor: Descriptor<DescriptorPublicKey>,
        assets: Assets<DescriptorPublicKey>,
    }

    impl Tester {
        pub fn new(secp: &Secp256k1<All>, desc_str: &str) -> Self {
            // let desc_str = "tr(xprv9uBuvtdjghkz8D1qzsSXS9Vs64mqrUnXqzNccj2xcvnCHPpXKYE1U2Gbh9CDHk8UPyF2VuXpVkDA7fk5ZP4Hd9KnhUmTscKmhee9Dp5sBMK)";
            let (descriptor, seckeys) =
                Descriptor::<DescriptorPublicKey>::parse_descriptor(secp, desc_str).unwrap();

            let assets = Assets {
                keys: seckeys.keys().cloned().collect(),
                ..Default::default()
            };

            Self { descriptor, assets }
        }

        pub fn gen_candidate(&self, derivation_index: u32, value: u64) -> TestCandidate {
            let descriptor = self.descriptor.at_derivation_index(derivation_index);
            let plan = descriptor.plan_satisfaction(&self.assets).unwrap();
            let txo = TxOut {
                value,
                script_pubkey: descriptor.script_pubkey(),
            };
            TestCandidate { txo, plan }
        }

        pub fn gen_weighted_value(&self, value: u64) -> WeightedValue {
            self.gen_candidate(0, value).into()
        }

        pub fn gen_weighted_values(&self, out: &mut Vec<WeightedValue>, count: usize, value: u64) {
            (0..count).for_each(|_| out.push(self.gen_candidate(0, value).into()))
        }

        pub fn gen_opts(&self, recipient_value: u64) -> CoinSelectorOpt {
            let recipient = self.gen_candidate(0, recipient_value);
            let drain = self.gen_candidate(0, 0);
            CoinSelectorOpt::fund_outputs(
                &[recipient.txo],
                &drain.txo,
                drain.plan.expected_weight() as u32,
            )
        }
    }
}

#[cfg(test)]
mod test_bnb {
    use bitcoin::secp256k1::Secp256k1;

    use crate::coin_select::{evaluate_cs::evaluate, ExcessStrategyKind};

    use super::{
        coin_select_bnb, tester::Tester, CoinSelector, CoinSelectorOpt, Vec, WeightedValue,
    };

    fn tester() -> Tester {
        const DESC_STR: &str = "tr(xprv9uBuvtdjghkz8D1qzsSXS9Vs64mqrUnXqzNccj2xcvnCHPpXKYE1U2Gbh9CDHk8UPyF2VuXpVkDA7fk5ZP4Hd9KnhUmTscKmhee9Dp5sBMK)";
        Tester::new(&Secp256k1::default(), DESC_STR)
    }

    #[test]
    fn not_enough_coins() {
        let t = tester();
        let candidates: Vec<WeightedValue> = vec![
            t.gen_candidate(0, 100_000).into(),
            t.gen_candidate(1, 100_000).into(),
        ];
        let opts = t.gen_opts(200_000);
        let mut selector = CoinSelector::new(&candidates, &opts);
        assert!(!coin_select_bnb(10_000, &mut selector));
    }

    #[test]
    fn exactly_enough_coins_preselected() {
        let t = tester();
        let candidates: Vec<WeightedValue> = vec![
            t.gen_candidate(0, 100_000).into(), // to preselect
            t.gen_candidate(1, 100_000).into(), // to preselect
            t.gen_candidate(2, 100_000).into(),
        ];
        let opts = CoinSelectorOpt {
            target_feerate: 0.0,
            ..t.gen_opts(200_000)
        };
        let selector = {
            let mut selector = CoinSelector::new(&candidates, &opts);
            selector.select(0); // preselect
            selector.select(1); // preselect
            selector
        };

        let evaluation =
            evaluate(selector, |cs| coin_select_bnb(10_000, cs)).expect("evaluation failed");
        println!("{}", evaluation);
        assert_eq!(evaluation.solution.selected, (0..=1).collect());
        assert_eq!(evaluation.solution.excess_strategies.len(), 1);
        assert_eq!(
            evaluation.feerate_offset(ExcessStrategyKind::ToFee).floor(),
            0.0
        );
    }

    /// `cost_of_change` acts as the upper-bound in Bnb, we check whether these boundaries are
    /// enforced in code
    #[test]
    fn cost_of_change() {
        let t = tester();
        let candidates: Vec<WeightedValue> = vec![
            t.gen_candidate(0, 200_000).into(),
            t.gen_candidate(1, 200_000).into(),
            t.gen_candidate(2, 200_000).into(),
        ];

        // lowest and highest possible `recipient_value` opts for derived `drain_waste`, assuming
        // that we want 2 candidates selected
        let (lowest_opts, highest_opts) = {
            let opts = t.gen_opts(0);

            let fee_from_inputs =
                (candidates[0].weight as f32 * opts.target_feerate).ceil() as u64 * 2;
            let fee_from_template =
                ((opts.base_weight + 2) as f32 * opts.target_feerate).ceil() as u64;

            let lowest_opts = CoinSelectorOpt {
                target_value: 400_000
                    - fee_from_inputs
                    - fee_from_template
                    - opts.drain_waste() as u64,
                ..opts
            };

            let highest_opts = CoinSelectorOpt {
                target_value: 400_000 - fee_from_inputs - fee_from_template,
                ..opts
            };

            (lowest_opts, highest_opts)
        };

        // test lowest possible target we are able to select
        let lowest_eval = evaluate(CoinSelector::new(&candidates, &lowest_opts), |cs| {
            coin_select_bnb(10_000, cs)
        });
        assert!(lowest_eval.is_ok());
        let lowest_eval = lowest_eval.unwrap();
        println!("LB {}", lowest_eval);
        assert_eq!(lowest_eval.solution.selected.len(), 2);
        assert_eq!(lowest_eval.solution.excess_strategies.len(), 1);
        assert_eq!(
            lowest_eval
                .feerate_offset(ExcessStrategyKind::ToFee)
                .floor(),
            0.0
        );

        // test highest possible target we are able to select
        let highest_eval = evaluate(CoinSelector::new(&candidates, &highest_opts), |cs| {
            coin_select_bnb(10_000, cs)
        });
        assert!(highest_eval.is_ok());
        let highest_eval = highest_eval.unwrap();
        println!("UB {}", highest_eval);
        assert_eq!(highest_eval.solution.selected.len(), 2);
        assert_eq!(highest_eval.solution.excess_strategies.len(), 1);
        assert_eq!(
            highest_eval
                .feerate_offset(ExcessStrategyKind::ToFee)
                .floor(),
            0.0
        );

        // test lower out of bounds
        let loob_opts = CoinSelectorOpt {
            target_value: lowest_opts.target_value - 1,
            ..lowest_opts
        };
        let loob_eval = evaluate(CoinSelector::new(&candidates, &loob_opts), |cs| {
            coin_select_bnb(10_000, cs)
        });
        assert!(loob_eval.is_err());
        println!("Lower OOB: {}", loob_eval.unwrap_err());

        // test upper out of bounds
        let uoob_opts = CoinSelectorOpt {
            target_value: highest_opts.target_value + 1,
            ..highest_opts
        };
        let uoob_eval = evaluate(CoinSelector::new(&candidates, &uoob_opts), |cs| {
            coin_select_bnb(10_000, cs)
        });
        assert!(uoob_eval.is_err());
        println!("Upper OOB: {}", uoob_eval.unwrap_err());
    }

    #[test]
    fn try_select() {
        let t = tester();
        let candidates: Vec<WeightedValue> = vec![
            t.gen_candidate(0, 300_000).into(),
            t.gen_candidate(1, 300_000).into(),
            t.gen_candidate(2, 300_000).into(),
            t.gen_candidate(3, 200_000).into(),
            t.gen_candidate(4, 200_000).into(),
        ];
        let make_opts = |v: u64| -> CoinSelectorOpt {
            CoinSelectorOpt {
                target_feerate: 0.0,
                ..t.gen_opts(v)
            }
        };

        let test_cases = vec![
            (make_opts(100_000), false, 0),
            (make_opts(200_000), true, 1),
            (make_opts(300_000), true, 1),
            (make_opts(500_000), true, 2),
            (make_opts(1_000_000), true, 4),
            (make_opts(1_200_000), false, 0),
            (make_opts(1_300_000), true, 5),
            (make_opts(1_400_000), false, 0),
        ];

        for (opts, expect_solution, expect_selected) in test_cases {
            let res = evaluate(CoinSelector::new(&candidates, &opts), |s| {
                coin_select_bnb(10_000, s)
            });
            assert_eq!(res.is_ok(), expect_solution);

            match res {
                Ok(eval) => {
                    println!("{}", eval);
                    assert_eq!(eval.feerate_offset(ExcessStrategyKind::ToFee), 0.0);
                    assert_eq!(eval.solution.selected.len(), expect_selected as _);
                }
                Err(err) => println!("expected failure: {}", err),
            }
        }
    }

    #[test]
    fn early_bailout_optimization() {
        let t = tester();

        // target: 300_000
        // candidates: 2x of 125_000, 1000x of 100_000, 1x of 50_000
        // expected solution: 2x 125_000, 1x 50_000
        // set bnb max tries: 1100, should succeed
        let candidates = {
            let mut candidates: Vec<WeightedValue> = vec![
                t.gen_candidate(0, 125_000).into(),
                t.gen_candidate(1, 125_000).into(),
                t.gen_candidate(2, 50_000).into(),
            ];
            (3..3 + 1000_u32)
                .for_each(|index| candidates.push(t.gen_candidate(index, 100_000).into()));
            candidates
        };
        let opts = CoinSelectorOpt {
            target_feerate: 0.0,
            ..t.gen_opts(300_000)
        };

        let result = evaluate(CoinSelector::new(&candidates, &opts), |cs| {
            coin_select_bnb(1100, cs)
        });
        assert!(result.is_ok());

        let eval = result.unwrap();
        println!("{}", eval);
        assert_eq!(eval.solution.selected, (0..=2).collect());
    }

    #[test]
    fn should_exhaust_iteration() {
        static MAX_TRIES: usize = 1000;
        let t = tester();
        let candidates = (0..MAX_TRIES + 1)
            .map(|index| t.gen_candidate(index as _, 10_000).into())
            .collect::<Vec<WeightedValue>>();
        let opts = t.gen_opts(10_001 * MAX_TRIES as u64);
        let result = evaluate(CoinSelector::new(&candidates, &opts), |cs| {
            coin_select_bnb(MAX_TRIES, cs)
        });
        assert!(result.is_err());
        println!("error as expected: {}", result.unwrap_err());
    }

    /// Solution should have fee >= min_absolute_fee
    #[test]
    fn min_absolute_fee() {
        let t = tester();
        let candidates = {
            let mut candidates = Vec::new();
            t.gen_weighted_values(&mut candidates, 5, 10_000);
            t.gen_weighted_values(&mut candidates, 5, 20_000);
            t.gen_weighted_values(&mut candidates, 5, 30_000);
            t.gen_weighted_values(&mut candidates, 10, 10_300);
            t.gen_weighted_values(&mut candidates, 10, 10_500);
            t.gen_weighted_values(&mut candidates, 10, 10_700);
            t.gen_weighted_values(&mut candidates, 10, 10_900);
            t.gen_weighted_values(&mut candidates, 10, 11_000);
            t.gen_weighted_values(&mut candidates, 10, 12_000);
            t.gen_weighted_values(&mut candidates, 10, 13_000);
            candidates
        };
        let mut opts = CoinSelectorOpt {
            min_absolute_fee: 1,
            ..t.gen_opts(100_000)
        };

        (1..=120_u64).for_each(|fee_factor| {
            opts.min_absolute_fee = fee_factor * 31;

            let result = evaluate(CoinSelector::new(&candidates, &opts), |cs| {
                coin_select_bnb(21_000, cs)
            });

            match result {
                Ok(result) => {
                    println!("Solution {}", result);
                    let fee = result.solution.excess_strategies[&ExcessStrategyKind::ToFee].fee;
                    assert!(fee >= opts.min_absolute_fee);
                    assert_eq!(result.solution.excess_strategies.len(), 1);
                }
                Err(err) => {
                    println!("No Solution: {}", err);
                }
            }
        });
    }

    /// TODO: UNIMPLEMENTED TESTS:
    /// * Decreasing feerate -> select less, increasing feerate -> select more
    /// * Excess strategies:
    ///     * We should always have `ExcessStrategy::ToFee`.
    ///     * We should only have `ExcessStrategy::ToRecipient` when `max_extra_target > 0`.
    ///     * We should only have `ExcessStrategy::ToDrain` when `drain_value >= min_drain_value`.
    /// * Fuzz
    ///     * Solution feerate should never be lower than target feerate
    ///     * Solution fee should never be lower than `min_absolute_fee`
    ///     * Preselected should always remain selected
    fn _todo() {}
}
