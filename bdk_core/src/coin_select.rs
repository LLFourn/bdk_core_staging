use crate::{
    collections::{BTreeSet, HashMap},
    Vec,
};
use bitcoin::{LockTime, Transaction, TxOut};

pub const TXIN_BASE_WEIGHT: u32 = (32 + 4 + 4) * 4;

#[derive(Debug, Clone)]
pub struct CoinSelector {
    candidates: Vec<WeightedValue>,
    selected: BTreeSet<usize>,
    opts: CoinSelectorOpt,
}

#[derive(Debug, Clone, Copy)]
pub struct WeightedValue {
    pub value: u64,
    /// Weight of including this `txin`: `prevout`, `nSequence`, `scriptSig` and `scriptWitness` are
    /// all included.
    pub weight: u32,
    /// Number of inputs; so we can calculate extra `varint` weight due to `vin` len changes.
    pub input_count: usize,
    pub is_segwit: bool,
}

impl WeightedValue {
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
        Self {
            target_value: 0,
            max_extra_target: 0,
            // 0.25 per wu i.e. 1 sat per byte
            target_feerate: 0.25,
            long_term_feerate: None,
            min_absolute_fee: 0,
            base_weight,
            drain_weight,
            spend_drain_weight,
            min_drain_value: 546, // this default is conservative (dust threshold for P2PKH)
        }
    }

    pub fn fund_outputs(txouts: &[TxOut], drain_output: &TxOut, spend_drain_weight: u32) -> Self {
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
            ..Self::from_weights(base_weight as u32, drain_weight as u32, spend_drain_weight)
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

impl CoinSelector {
    pub fn candidates(&self) -> &[WeightedValue] {
        &self.candidates
    }

    pub fn candidate(&self, index: usize) -> &WeightedValue {
        &self.candidates[index]
    }

    pub fn new(candidates: Vec<WeightedValue>, opts: CoinSelectorOpt) -> Self {
        Self {
            candidates,
            selected: Default::default(),
            opts,
        }
    }

    pub fn opts(&self) -> CoinSelectorOpt {
        self.opts
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        self.selected.insert(index);
    }

    pub fn deselect(&mut self, index: usize) {
        self.selected.remove(&index);
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected.contains(&index)
    }

    pub fn is_empty(&self) -> bool {
        self.selected.is_empty()
    }

    /// Weight sum of all selected inputs.
    pub fn selected_weight(&self) -> u32 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].weight)
            .sum()
    }

    /// Effective value sum of all selected inputs.
    pub fn selected_effective_value(&self) -> i64 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].effective_value(&self.opts))
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
            .selected()
            .find(|(_, wv)| wv.is_segwit)
            .map(|_| 2)
            .unwrap_or(0);
        let vin_count_varint_extra_weight = {
            let input_count = self.selected().map(|(_, wv)| wv.input_count).sum::<usize>();
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

    pub fn selected(&self) -> impl Iterator<Item = (usize, WeightedValue)> + '_ {
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

    pub fn selected_value(&self) -> u64 {
        self.selected().map(|(_, wv)| wv.value).sum::<u64>()
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let weight_without_drain = self.current_weight();
        let weight_with_drain = weight_without_drain + self.opts.drain_weight;

        let fee_without_drain =
            (weight_without_drain as f32 * self.opts.target_feerate).ceil() as u64;
        let fee_with_drain = (weight_with_drain as f32 * self.opts.target_feerate).ceil() as u64;

        let inputs_minus_outputs = {
            let target_value = self.opts.target_value;
            let selected = self.selected_value();

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
            selected: self.selected.clone(),
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
    /// returns feerate in sats/wu
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
    let opts = selection.opts();

    let base_weight = {
        let (has_segwit, max_input_count) = selection
            .candidates()
            .iter()
            .fold((false, 0_usize), |(is_segwit, input_count), c| {
                (is_segwit || c.is_segwit, input_count + c.input_count)
            });

        selection.selected_weight()
            + opts.base_weight
            + if has_segwit { 2_u32 } else { 0_u32 }
            + (varint_size(max_input_count) - 1) * 4
    };

    let pool = {
        // TODO: Another optimisation we could do is figure out candidate with smallest waste, and
        // if we find a result with waste equal to this, we can just break.
        let mut pool = selection
            .unselected()
            .into_iter()
            .filter(|&index| selection.candidate(index).effective_value(&opts) > 0)
            .collect::<Vec<_>>();
        // sort by descending effective value
        pool.sort_unstable_by(|&a, &b| {
            let a = selection.candidate(a).effective_value(&opts);
            let b = selection.candidate(b).effective_value(&opts);
            b.cmp(&a)
        });
        pool
    };

    let mut pos = 0_usize;

    let target_value =
        opts.target_value as i64 + (base_weight as f32 * opts.target_feerate).ceil() as i64;
    let mut remaining_value = pool
        .iter()
        .map(|&index| selection.candidate(index).effective_value(&opts))
        .sum::<i64>();

    if remaining_value < target_value {
        return false;
    }

    let abs_target_value = opts.target_value + opts.min_absolute_fee;
    let mut abs_remaining_value = pool
        .iter()
        .map(|&i| selection.candidate(i).value)
        .sum::<u64>();

    if abs_remaining_value < abs_target_value {
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
        let abs_current_value = selection.selected().map(|(_, c)| c.value).sum::<u64>();
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
                #[cfg(feature = "std")]
                println!("solution @ try {} with waste {}", try_index, current_waste);
                best_selection.replace(selection.clone());
            }
            true
        } else {
            false
        };

        if backtrack {
            // println!("backtrack @ try {}", try_index);
            let last_selected_pos = (0..pos).rev().find(|&pos| {
                let is_selected = selection.is_selected(pool[pos]);
                if !is_selected {
                    let candidate = &selection.candidate(pool[pos]);
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

        let candidate = selection.candidate(pool[pos]);
        remaining_value -= candidate.effective_value(&opts);
        abs_remaining_value -= candidate.value;

        // if the candidate at the previous position is NOT selected and has the same weight and
        // value as the current candidate, we skip the current candidate
        if !selection.is_empty() {
            let prev_candidate = selection.candidate(pool[pos - 1]);
            if !selection.is_selected(pool[pos - 1])
                && candidate.value == prev_candidate.value
                && candidate.weight == prev_candidate.weight
            {
                // println!("skipped @ try {}", try_index);
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

#[cfg(test)]
mod test {
    use super::*;
    use alloc::vec::Vec;
    use bitcoin::{
        secp256k1::{All, Secp256k1},
        TxOut,
    };
    use miniscript::{
        plan::{Assets, Plan},
        Descriptor, DescriptorPublicKey,
    };
    use rand::Rng;
    pub struct TestDescriptors {
        descriptors: Vec<Descriptor<DescriptorPublicKey>>,
        assets: Assets<DescriptorPublicKey>,
    }

    impl TestDescriptors {
        pub fn new(secp: &Secp256k1<All>) -> Self {
            let tr_str = "tr(xprv9uBuvtdjghkz8D1qzsSXS9Vs64mqrUnXqzNccj2xcvnCHPpXKYE1U2Gbh9CDHk8UPyF2VuXpVkDA7fk5ZP4Hd9KnhUmTscKmhee9Dp5sBMK)";
            let (tr_desc, tr_sks) =
                Descriptor::<DescriptorPublicKey>::parse_descriptor(secp, tr_str).unwrap();

            let assets = Assets {
                keys: tr_sks.keys().cloned().collect(),
                ..Default::default()
            };

            let descriptors = vec![tr_desc];
            Self {
                descriptors,
                assets,
            }
        }

        pub fn generate_candidate(&self, min: u64, max: u64) -> (Plan<DescriptorPublicKey>, TxOut) {
            let mut rng = rand::thread_rng();
            let desc_index = rng.gen_range(0_usize..self.descriptors.len());
            let desc = self.descriptors[desc_index].at_derivation_index(0);
            let plan = desc.plan_satisfaction(&self.assets).unwrap();
            let value = rng.gen_range(min..max);

            let txo = TxOut {
                value,
                script_pubkey: desc.script_pubkey(),
            };

            (plan, txo)
        }

        pub fn generate_candidates(
            &self,
            count: usize,
            min: u64,
            max: u64,
        ) -> Vec<(Plan<DescriptorPublicKey>, TxOut)> {
            (0..count)
                .map(|_| self.generate_candidate(min, max))
                .collect()
        }
    }

    #[test]
    fn test_bnb() {
        let secp = Secp256k1::default();
        let test_desc = TestDescriptors::new(&secp);

        (0..1).for_each(|_| {
            println!("-----");
            let mut candidates = test_desc.generate_candidates(2100, 10_000, 100_000);
            let (_, mut recipient_txo) = candidates.pop().unwrap();
            recipient_txo.value = 213_123;
            let (drain_plan, drain_txo) = candidates.pop().unwrap();

            let cs_opts = CoinSelectorOpt {
                target_feerate: 1.0,
                long_term_feerate: Some(0.25),
                // min_absolute_fee: 1200,
                // max_extra_target: 1000,
                ..CoinSelectorOpt::fund_outputs(
                    &[recipient_txo.clone()],
                    &drain_txo,
                    TXIN_BASE_WEIGHT + drain_plan.expected_weight() as u32,
                )
            };

            println!("cs_opts: {:#?}", cs_opts);

            let cs_candidates = candidates
                .iter()
                .map(|(plan, txo)| WeightedValue {
                    value: txo.value,
                    weight: TXIN_BASE_WEIGHT + plan.expected_weight() as u32,
                    input_count: 1,
                    is_segwit: plan.witness_version().is_some(),
                })
                .collect::<Vec<_>>();

            let mut selection = CoinSelector::new(cs_candidates, cs_opts);
            if coin_select_bnb(21_000, &mut selection) {
                let results = selection
                    .finish()
                    .expect("bnb returned true so finish should succeed");
                println!("result: {:#?}", results);
                for (strat, meta) in results.excess_strategies {
                    let feerate = meta.feerate();
                    assert!(feerate >= cs_opts.target_feerate, "feerate undershot");
                    println!("{}: feerate: {} sats/wu", strat, feerate);
                }
            } else {
                println!("no bnb result!");
            }
        })
    }
}
