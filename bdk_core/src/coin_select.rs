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
    pub is_segwit: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// The value we need to select.
    pub target_value: u64,
    /// The feerate we should try and achieve in sats per weight unit.
    pub target_feerate: f32,
    /// TODO
    pub long_term_feerate: Option<f32>,
    /// The minimum absolute fee.
    pub min_absolute_fee: u64,
    /// The weight of the template transaction including fixed inputs and outputs.
    pub base_weight: u32,
    /// The weight of the drain (change) output.
    pub drain_weight: u32,
    /// TODO
    pub spend_drain_weight: u32,
}

impl CoinSelectorOpt {
    pub fn from_weights(base_weight: u32, drain_weight: u32, spend_drain_weight: u32) -> Self {
        Self {
            target_value: 0,
            // 0.25 per wu i.e. 1 sat per byte
            target_feerate: 0.25,
            long_term_feerate: None,
            min_absolute_fee: 0,
            base_weight,
            drain_weight,
            spend_drain_weight,
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
        self.opts.base_weight + self.input_weight() + witness_header_extra_weight
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

        let fee_with_drain = ((self.opts.target_feerate * weight_with_drain as f32).ceil() as u64)
            .max(self.opts.min_absolute_fee);
        let fee_without_drain = ((self.opts.target_feerate * weight_without_drain as f32).ceil()
            as u64)
            .max(self.opts.min_absolute_fee);

        let inputs_minus_outputs = {
            let inputs_minus_outputs = self.selected_value() as i64 - self.opts.target_value as i64;
            if inputs_minus_outputs < fee_without_drain as i64 {
                return Err(SelectionFailure::InsufficientFunds {
                    selected: self.selected_value(),
                    needed: fee_without_drain + self.opts.target_value,
                });
            }
            // if inputs_minus_outputs < self.opts.min_absolute_fee as i64 {
            //     return Err(SelectionFailure::AbsoluteFeeTooLow {
            //         needed: self.opts.min_absolute_fee,
            //         had: inputs_minus_outputs as u64,
            //     });
            // }
            inputs_minus_outputs as u64
        };

        // // check fee rate satisfied
        // let feerate_without_drain = inputs_minus_outputs as f32 / weight_without_drain as f32;
        // // we simply don't have enough fee to achieve the feerate
        // if feerate_without_drain < self.opts.target_feerate {
        //     return Err(SelectionFailure::FeerateTooLow {
        //         needed: self.opts.target_feerate,
        //         had: feerate_without_drain,
        //     });
        // }

        let drain_invalid = inputs_minus_outputs < fee_with_drain; // TODO: We need minimum drain value

        let excess_without_drain = inputs_minus_outputs - fee_without_drain;
        let drain_value = inputs_minus_outputs.saturating_sub(fee_with_drain);

        // waste calculations
        let input_waste = self.waste_of_inputs();
        let waste_without_drain = input_waste + excess_without_drain as i64;
        let waste_with_drain = input_waste + self.opts.drain_waste();

        Ok(Selection {
            selected: self.selected.clone(),
            excess_without_drain,
            drain_value,
            drain_invalid,
            fee_without_drain,
            fee_with_drain,
            weight_without_drain,
            weight_with_drain,
            waste_without_drain,
            waste_with_drain,
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
    pub excess_without_drain: u64, // excess = input values - target_value - fee
    pub drain_value: u64,
    pub drain_invalid: bool,

    pub fee_without_drain: u64, // fee = input values - target_value - excess || fee = total_weight * target_feerate
    pub fee_with_drain: u64,    //

    pub weight_without_drain: u32, // weight without drain
    pub weight_with_drain: u32,    // base_weight + drain_weight

    pub waste_without_drain: i64, // waste of the inputs only (without drain: + excess, with drain: + cost_of_drain)
    pub waste_with_drain: i64,
}

impl Selection {
    pub fn apply_selection<'a, T>(
        &'a self,
        candidates: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        self.selected.iter().map(|i| &candidates[*i])
    }
}
