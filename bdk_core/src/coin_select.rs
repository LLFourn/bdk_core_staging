use crate::{collections::BTreeSet, Vec};
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
    pub fn from_weights(base_weight: u32, drain_weight: u32, spend_drain_weight: u32) -> Self {
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

    pub fn drain_waste(&self) -> f32 {
        self.drain_weight as f32 * self.target_feerate
            + self.spend_drain_weight as f32 * self.long_term_feerate.unwrap_or(self.target_feerate)
    }
}

impl CoinSelector {
    pub fn candidates(&self) -> &[WeightedValue] {
        &self.candidates
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

    /// Weight of all inputs.
    pub fn input_weight(&self) -> u32 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].weight)
            .sum()
    }

    pub fn waste_of_inputs(&self) -> f32 {
        self.input_weight() as f32
            * (self.opts.target_feerate
                - self
                    .opts
                    .long_term_feerate
                    .unwrap_or(self.opts.target_feerate))
    }

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
            + self.input_weight()
            + witness_header_extra_weight
            + vin_count_varint_extra_weight
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

    pub fn select_until_finished(&mut self) -> Result<Vec<Selection>, SelectionFailure> {
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

    pub fn finish(&self) -> Result<Vec<Selection>, SelectionFailure> {
        let weight_without_drain = self.current_weight();
        let weight_with_drain = weight_without_drain + self.opts.drain_weight;

        let fee_with_drain = ((self.opts.target_feerate * weight_with_drain as f32).ceil() as u64)
            .max(self.opts.min_absolute_fee);
        let fee_without_drain = ((self.opts.target_feerate * weight_without_drain as f32).ceil()
            as u64)
            .max(self.opts.min_absolute_fee);

        let inputs_minus_outputs = {
            let target_value = self.opts.target_value as i64;
            let selected_value = self.selected_value() as i64;

            let (missing, unsatisfied) = SelectionConstraint::unsatisfied(
                target_value,
                selected_value,
                fee_without_drain as i64,
                self.opts.min_absolute_fee as i64,
            );

            if !unsatisfied.is_empty() {
                return Err(SelectionFailure::InsufficientFunds {
                    selected: selected_value as _,
                    missing,
                    unsatisfied,
                });
            }

            (selected_value - target_value) as u64
        };

        let drain_valid = inputs_minus_outputs >= fee_with_drain + self.opts.min_drain_value;
        let excess_without_drain = inputs_minus_outputs - fee_without_drain;
        let input_waste = self.waste_of_inputs();

        // prepare results
        let mut results = Vec::with_capacity(3);

        // no drain, excess to fee
        results.push(Selection {
            selected: self.selected.clone(),
            excess: excess_without_drain,
            drain: None,
            recipient_value: self.opts.target_value,
            fee: fee_without_drain + excess_without_drain,
            weight: weight_without_drain,
            waste: input_waste + excess_without_drain as f32,
        });

        // no drain, excess to recipient
        // if `excess == 0`, this result will be the same as the previous, so we don't consider it
        if excess_without_drain > 0 && excess_without_drain <= self.opts.max_extra_target {
            results.push(Selection {
                selected: self.selected.clone(),
                excess: excess_without_drain,
                drain: None,
                recipient_value: self.opts.target_value + excess_without_drain,
                fee: fee_without_drain,
                weight: weight_without_drain,
                waste: input_waste,
            });
        }

        // with drain
        if drain_valid {
            let drain_value = inputs_minus_outputs.saturating_sub(fee_with_drain);

            results.push(Selection {
                selected: self.selected.clone(),
                excess: excess_without_drain,
                drain: Some(drain_value),
                recipient_value: self.opts.target_value,
                fee: fee_with_drain,
                weight: weight_with_drain,
                waste: input_waste + self.opts.drain_waste(),
            });
        }

        // sort by ascending waste
        results.sort_unstable_by(|a, b| a.waste.partial_cmp(&b.waste).unwrap());
        Ok(results)
    }
}

#[derive(Clone, Debug)]
pub enum SelectionFailure {
    InsufficientFunds {
        selected: u64,
        missing: u64,
        unsatisfied: Vec<SelectionConstraint>,
    },
}

impl core::fmt::Display for SelectionFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectionFailure::InsufficientFunds {
                selected,
                missing,
                unsatisfied,
            } => write!(
                f,
                "insufficient coins selected; selected={}, missing={}, unsatisfied={:?}",
                selected, missing, unsatisfied
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

impl SelectionConstraint {
    /// Returns the value required to satisfy all constraints and a vec of unsatisfied contraints.
    pub fn unsatisfied(
        target_value: i64,
        selected_value: i64,
        fee_without_drain: i64,
        min_absolute_fee: i64,
    ) -> (u64, Vec<Self>) {
        let mut unsatisfied = Vec::with_capacity(3);
        let mut remaining = 0_i64;

        let mut update = |c: Self, v: i64| {
            if v < 0 {
                unsatisfied.push(c);
                let v = v.abs();
                if v > remaining {
                    remaining = v;
                }
            }
        };

        let target_value_surplus = selected_value - target_value;
        let target_fee_surplus = target_value_surplus - fee_without_drain;
        let min_absolute_fee_surplus = target_value_surplus - min_absolute_fee;

        update(Self::TargetValue, target_value_surplus);
        update(Self::TargetFee, target_fee_surplus);
        update(Self::MinAbsoluteFee, min_absolute_fee_surplus);

        (remaining as u64, unsatisfied)
    }
}

#[derive(Clone, Debug)]
pub struct Selection {
    pub selected: BTreeSet<usize>,
    pub excess: u64,
    pub drain: Option<u64>,
    pub recipient_value: u64,
    pub fee: u64,
    pub weight: u32,
    pub waste: f32,
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

pub fn coin_select_bnb(
    max_tries: usize,
    selection: &mut CoinSelector,
) -> Result<Vec<Selection>, SelectionFailure> {
    fn effective_value(candidate: &WeightedValue, opts: &CoinSelectorOpt) -> f32 {
        candidate.value as f32 - candidate.weight as f32 * opts.target_feerate
    }

    fn selected_value(selection: &CoinSelector) -> f32 {
        selection
            .selected()
            .map(|(_, c)| effective_value(&c, &selection.opts()))
            .sum()
    }

    fn is_selected(selection: &CoinSelector, index: usize) -> bool {
        selection.selected().find(|(i, _)| *i == index).is_some()
    }

    let opts = selection.opts();

    let target_value = {
        let has_segwit = selection
            .candidates()
            .iter()
            .find(|c| c.is_segwit)
            .is_some();

        let max_input_count = selection
            .candidates()
            .iter()
            .map(|c| c.input_count)
            .sum::<usize>();

        let base_weight = selection.input_weight()
            + opts.base_weight
            + if has_segwit { 2_u32 } else { 0_u32 }
            + (varint_size(max_input_count) - 1) * 4;

        opts.target_value as f32 + base_weight as f32 * opts.target_feerate
    };

    let pool = {
        // TODO: Another optimisation we could do is figure out candidate with smallest waste, and
        // if we find a result with waste equal to this, we can just break.
        let mut pool = selection
            .unselected()
            .into_iter()
            .filter(|&index| effective_value(&selection.candidates()[index], &opts) > 0.0)
            .collect::<Vec<_>>();
        // sort by descending effective value
        pool.sort_unstable_by(|&a, &b| {
            let a = effective_value(&selection.candidates()[a], &opts);
            let b = effective_value(&selection.candidates()[b], &opts);
            b.partial_cmp(&a).expect("failed to compare")
        });
        pool
    };
    let mut pos = 0_usize;

    let mut remaining_value = pool
        .iter()
        .map(|&index| effective_value(&selection.candidates()[index], &opts))
        .sum::<f32>();

    if remaining_value <= target_value {
        todo!("what is a good way of handling this error?");
    }

    let drain_cost = opts.drain_weight as f32 * opts.target_feerate
        + opts.spend_drain_weight as f32 * opts.long_term_feerate.unwrap_or(opts.target_feerate);
    let feerate_decreasing =
        opts.target_feerate > opts.long_term_feerate.unwrap_or(opts.target_feerate);
    let upper_bound = target_value + drain_cost + opts.max_extra_target as f32;

    let mut best_selection = Option::<CoinSelector>::None;

    for try_index in 0..max_tries {
        if try_index > 0 {
            pos += 1;
        }

        let backtrack = {
            let current_value = selected_value(selection);
            let current_input_waste = selection.waste_of_inputs();

            let best_waste = best_selection
                .as_ref()
                .map(|b| b.waste_of_inputs() + selected_value(b) - target_value)
                .unwrap_or(f32::MAX);

            if current_value + remaining_value < target_value || current_value > upper_bound {
                // remaining value is not enough to reach target OR current value surpasses upper bound
                true
            } else if feerate_decreasing && current_input_waste > best_waste {
                // when feerate_decreasing, waste is guaranteed to increase with each selection,
                // so backtrack when we have already surpassed best waste
                true
            } else if current_value >= target_value {
                // we have found a solution, is the current waste better than our best?
                let current_waste = current_input_waste + current_value - target_value;
                if current_waste <= best_waste {
                    #[cfg(feature = "std")]
                    println!("solution @ try {} with waste {}", try_index, current_waste);
                    best_selection.replace(selection.clone());
                }
                true
            } else {
                false
            }
        };

        if backtrack {
            let last_selected_pos = (0..pos).rev().find(|&pos| {
                let is_selected = is_selected(selection, pool[pos]);
                if !is_selected {
                    remaining_value += effective_value(&selection.candidates()[pool[pos]], &opts);
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

        let candidate = selection.candidates()[pool[pos]];
        remaining_value -= effective_value(&candidate, &opts);

        // if the candidate at the previous position is NOT selected and has the same weight and
        // value as the current candidate, we skip the current candidate
        if pos > 0 {
            let prev_candidate = selection.candidates()[pool[pos - 1]];
            if !is_selected(selection, pool[pos - 1])
                && candidate.value == prev_candidate.value
                && candidate.weight == prev_candidate.weight
            {
                // println!("skipped @ try {}", try_index);
                continue;
            }
        }

        selection.select(pool[pos]);
    }

    if let Some(best_selection) = best_selection {
        *selection = best_selection;
    }
    selection.finish()
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

        let mut candidates = test_desc.generate_candidates(100, 1000, 10_000);
        let (_, mut recipient_txo) = candidates.pop().unwrap();
        recipient_txo.value = 110_000;
        let (drain_plan, drain_txo) = candidates.pop().unwrap();

        let cs_opts = CoinSelectorOpt {
            target_feerate: 0.26,
            long_term_feerate: Some(0.25),
            // min_absolute_fee: 1000,
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
        match coin_select_bnb(100_000, &mut selection) {
            Ok(results) => {
                println!("results: {:#?}", results);
                for res in results {
                    let feerate = res.fee as f32 / res.weight as f32;
                    assert!(feerate >= cs_opts.target_feerate, "feerate undershot");
                    println!("feerate: {} sats/wu", feerate);
                }
            }
            Err(err) => println!("err: {:#?}", err),
        };
    }
}
