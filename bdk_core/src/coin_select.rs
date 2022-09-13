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

    pub fn drain_waste(&self) -> i64 {
        (self.drain_weight as f32 * self.target_feerate
            + self.spend_drain_weight as f32
                * self.long_term_feerate.unwrap_or(self.target_feerate)) as i64
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

    /// Weight of all inputs.
    pub fn input_weight(&self) -> u32 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].weight)
            .sum()
    }

    pub fn waste_of_inputs(&self) -> i64 {
        (self.input_weight() as f32
            * (self.opts.target_feerate
                - self
                    .opts
                    .long_term_feerate
                    .unwrap_or(self.opts.target_feerate))) as i64
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
            waste: input_waste + excess_without_drain as i64,
        });

        // no drain, excess to recipient
        if excess_without_drain <= self.opts.max_extra_target {
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
        results.sort_unstable_by_key(|s| s.waste);
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
