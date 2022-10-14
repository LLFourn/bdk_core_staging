use core::borrow::Borrow;

use bitcoin::{LockTime, Transaction, TxOut};

use super::*;

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

    /// Effective value of this input candidate: `actual_value - input_weight * feerate (sats/wu)`.
    pub fn effective_value(&self, effective_feerate: f32) -> i64 {
        // We prefer undershooting the candidate's effective value (so we over estimate the fee of a
        // candidate). If we overshoot the candidate's effective value, it may be possible to find a
        // solution which does not meet the target feerate.
        self.value as i64 - (self.weight as f32 * effective_feerate).ceil() as i64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// The value we need to select.
    /// If the value is `None` then the selection will be complete if it can pay for the drain
    /// output and satisfy the other constraints (e.g. minimum fees).
    pub target_value: u64,
    /// The weight of the template transaction including fixed fields and outputs.
    pub base_weight: u32,
}

impl CoinSelectorOpt {
    // TODO: we need to know number of outputs to take into account varint
    pub fn new(base_weight: u32, target_value: u64) -> Self {
        Self {
            target_value,
            base_weight,
        }
    }

    pub fn fund_outputs(txouts: &[TxOut]) -> Self {
        let tx = Transaction {
            input: vec![],
            version: 1,
            lock_time: LockTime::ZERO.into(),
            output: txouts.to_vec(),
        };
        let base_weight = tx.weight();
        Self::new(
            base_weight as u32,
            txouts.iter().map(|txout| txout.value).sum(),
        )
    }
}

/// [`CoinSelector`] is responsible for selecting and deselecting from a set of canididates.
#[derive(Debug)]
pub struct CoinSelector<'a, S> {
    pub opts: CoinSelectorOpt,
    candidates: &'a [WeightedValue],
    selected: BTreeSet<usize>,
    state: core::marker::PhantomData<S>,
}

#[derive(Debug, Clone, Copy)]
pub struct Finished;
#[derive(Debug, Clone, Copy)]
pub struct Unfinished;

impl<'a> CoinSelector<'a, Finished> {
    pub fn apply_selection<T>(&self, candidates: &'a [T]) -> impl Iterator<Item = &'a T> + '_ {
        self.selected.iter().map(|i| &candidates[*i])
    }
}

impl<'a> CoinSelector<'a, Unfinished> {
    pub fn deselect(&mut self, index: usize) -> bool {
        self.selected.remove(&index)
    }

    pub fn new(candidates: &'a [WeightedValue], opts: CoinSelectorOpt) -> Self {
        Self {
            candidates,
            selected: Default::default(),
            opts,
            state: core::marker::PhantomData,
        }
    }
}

impl<'a, S> CoinSelector<'a, S> {
    pub fn candidate(&self, index: usize) -> &WeightedValue {
        &self.candidates[index]
    }

    pub fn select(&mut self, index: usize) -> bool {
        assert!(index < self.candidates.len());
        self.selected.insert(index)
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

    /// Absolute value sum of all selected inputs.
    pub fn selected_value(&self) -> u64 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].value)
            .sum()
    }

    /// Current weight of template tx + selected inputs.
    pub fn current_weight(&self) -> u32 {
        // TODO: take into account the witness stack length for each input
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

    pub fn effective_value(&self, feerate: f32) -> i64 {
        self.selected_value() as i64 - self.fee_needed(feerate) as i64
    }

    pub fn effective_value_with_drain(&self, feerate: f32, drain_weight: u32) -> i64 {
        self.selected_value() as i64 - self.fee_needed_with_drain(feerate, drain_weight) as i64
    }

    pub fn rate_excess(&self, feerate: f32) -> i64 {
        self.effective_value(feerate) - self.opts.target_value as i64
    }

    pub fn rate_excess_with_drain(&self, feerate: f32, drain_weight: u32) -> i64 {
        self.effective_value_with_drain(feerate, drain_weight) - self.opts.target_value as i64
    }

    pub fn fee_needed(&self, feerate: f32) -> u64 {
        (self.current_weight() as f32 * feerate).ceil() as u64
    }

    pub fn fee_needed_with_drain(&self, feerate: f32, drain_weight: u32) -> u64 {
        (self.weight_with_drain(drain_weight) as f32 * feerate).ceil() as u64
    }

    pub fn weight_with_drain(&self, drain_weight: u32) -> u32 {
        self.current_weight() + drain_weight
    }

    pub fn abs_excess(&self, abs_fee: u64) -> i64 {
        self.selected_value() as i64 - self.opts.target_value as i64 - abs_fee as i64
    }

    pub fn excess(&self, min_fee: u64, feerate: f32) -> i64 {
        self.rate_excess(feerate).min(self.abs_excess(min_fee))
    }

    /// The value that can be drained to a change output while maintaining the `feerate` and the `min_fee`
    pub fn drain_excess(&self, min_fee: u64, feerate: f32, drain_weight: u32) -> i64 {
        self.rate_excess_with_drain(feerate, drain_weight)
            .min(self.abs_excess(min_fee))
    }

    // /// Waste sum of all selected inputs.
    pub fn selected_waste(&self, feerate: f32, long_term_feerate: f32) -> f32 {
        self.selected_weight() as f32 * (feerate - long_term_feerate)
    }

    pub fn waste(
        &self,
        min_fee: u64,
        feerate: f32,
        long_term_feerate: f32,
        drain_weights: Option<(u32, u32)>,
    ) -> f32 {
        let mut waste = self.selected_waste(feerate, long_term_feerate);
        match drain_weights {
            Some((drain_weight, drain_spend_weight)) => {
                waste +=
                    drain_weight as f32 * feerate + drain_spend_weight as f32 * long_term_feerate;
            }
            None => waste += self.excess(min_fee, feerate) as f32,
        }

        waste
    }

    // TODO: Ask even what this is for and whether we need it
    // pub fn effective_target(&self) -> i64 {
    //     let (has_segwit, max_input_count) = self
    //         .candidates
    //         .iter()
    //         .fold((false, 0_usize), |(is_segwit, input_count), c| {
    //             (is_segwit || c.is_segwit, input_count + c.input_count)
    //         });

    //     let effective_base_weight = self.opts.base_weight
    //         + if has_segwit { 2_u32 } else { 0_u32 }
    //         + (varint_size(max_input_count) - 1) * 4;

    //     self.opts.target_value.unwrap_or(0) as i64
    //         + (effective_base_weight as f32 * self.opts.target_feerate).ceil() as i64
    // }

    pub fn selected(&self) -> impl ExactSizeIterator<Item = (usize, &'a WeightedValue)> + '_ {
        self.selected
            .iter()
            .map(|&index| (index, &self.candidates[index]))
    }

    pub fn unselected(&self) -> impl Iterator<Item = (usize, &'a WeightedValue)> + '_ {
        self.candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.selected.contains(index))
    }

    pub fn selected_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        self.selected.iter().cloned()
    }

    pub fn unselected_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.candidates.len()).filter(|index| !self.selected.contains(index))
    }

    pub fn all_selected(&self) -> bool {
        self.selected.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        self.selected = (0..self.candidates.len()).collect();
    }

    pub fn iter_finished(
        &self,
        candidates: impl IntoIterator<Item = impl Borrow<usize>>,
    ) -> impl Iterator<Item = CoinSelector<'a, Finished>> {
        let mut selector = self.clone();

        candidates.into_iter().filter_map(move |index| {
            selector.select(*index.borrow());
            selector.finish()
        })
    }

    pub fn finish(&self) -> Option<CoinSelector<'a, Finished>> {
        if !self.selected.is_empty() && self.selected_value() >= self.opts.target_value {
            Some(CoinSelector {
                state: core::marker::PhantomData,
                opts: self.opts,
                candidates: self.candidates,
                selected: self.selected.clone(),
            })
        } else {
            None
        }
    }

    pub fn unfinish(self) -> CoinSelector<'a, Unfinished> {
        CoinSelector {
            opts: self.opts,
            candidates: self.candidates,
            selected: self.selected,
            state: core::marker::PhantomData,
        }
    }

    pub fn branch_and_bound<O, F, G, H>(
        &self,
        score_fn: F,
        heuristic_fn: G,
    ) -> impl Iterator<Item = Option<(CoinSelector<'a, Finished>, O)>>
    where
        O: Ord + core::fmt::Debug + Clone,
        F: FnMut(&CoinSelector<'a, Finished>, H) -> Option<O>,
        G: FnMut(&CoinSelector<'a, Unfinished>, &[usize]) -> Option<(O, H)>,
    {
        crate::coin_select::bnb2::BnbIter::new(self, score_fn, heuristic_fn)
    }

    pub fn minimize_waste<'b, C>(
        &'b self,
        feerate: f32,
        min_fee: u64,
        long_term_feerate: f32,
        mut change_policy: C,
    ) -> impl Iterator<Item = Option<(CoinSelector<'a, Finished>, u32)>> + 'b
    where
        C: FnMut(&CoinSelector<'a, Finished>) -> Option<(u32, u32)> + 'b,
    {
        let rate_diff = feerate - long_term_feerate;

        let score_fn = move |cs: &CoinSelector<'a, Finished>, drain_weights: Option<(u32, u32)>| {
            let excess = cs.excess(min_fee, feerate);
            if excess < 0 {
                return None;
            }

            let score = cs
                .waste(min_fee, feerate, long_term_feerate, drain_weights)
                .ceil() as u32;
            Some(score)
        };

        let heuristic = move |cs: &CoinSelector<'a, Unfinished>, remaining: &[usize]| {
            let mut cs = cs.clone();
            let finish_now = self.finish();
            // NOTE: This logic in this function is implicitly assuming that if in our current state we'd have a
            // change output, adding any new inputs would also result in a change output.
            // This assumption helps a lot with branching as it demotes any selection after a change is added.
            let drain_weights = finish_now.as_ref().and_then(&mut change_policy);
            let representative = if rate_diff >= 0.0 {
                // If feerate >= long_term_feerate then the least waste we can possibly have is the
                // waste of what is currently selected + whatever we need to finish.
                // NOTE: this assumes remaining are sorted in descending effective value.
                finish_now.or_else(|| {
                    remaining
                        .iter()
                        .filter_map(|i| {
                            cs.select(*i);
                            cs.finish()
                        })
                        .find(|cs| cs.excess(min_fee, feerate) >= 0)
                })?
            } else {
                // if the feerate < long_term_feerate then selecting everything remaining gives the
                // lower bound on this selection's waste
                for i in remaining {
                    cs.select(*i);
                }
                let cs = cs.finish()?;
                if cs.excess(min_fee, feerate) < 0 {
                    return None;
                }
                cs
            };

            let lower_bound_score = representative
                .waste(min_fee, feerate, long_term_feerate, drain_weights)
                .ceil() as u32;
            // we provide this hint to score function so it doesn't have to call change_policy again.
            let hint = drain_weights;
            Some((lower_bound_score, hint))
        };

        crate::coin_select::bnb2::BnbIter::new(self, score_fn, heuristic)
    }
}

impl<'a, S> Clone for CoinSelector<'a, S> {
    fn clone(&self) -> Self {
        Self {
            selected: self.selected.clone(),
            ..*self
        }
    }
}

#[derive(Clone, Debug)]
pub struct InsufficientFunds {
    selected: u64,
    missing: u64,
}

impl core::fmt::Display for InsufficientFunds {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InsufficientFunds { selected, missing } => write!(
                f,
                "insufficient coins selected; selected={}, missing={}",
                selected, missing
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for InsufficientFunds {}

// #[derive(Clone, Debug)]
// pub struct Selection {
//     pub selected: BTreeSet<usize>,
//     pub excess: u64,
//     pub recipient_value: Option<u64>,
//     pub drain_value: Option<u64>,
//     pub fee: u64,
//     pub weight: u32,
// }

// impl Selection {
//     pub fn apply_selection<'a, T>(
//         &'a self,
//         candidates: &'a [T],
//     ) -> impl Iterator<Item = &'a T> + 'a {
//         self.selected.iter().map(|i| &candidates[*i])
//     }
// }

// #[cfg(test)]
// mod test {
//     use crate::coin_select::{ExcessStrategyKind, SelectionConstraint};

//     use super::{CoinSelector, CoinSelectorOpt, WeightedValue};

//     /// Ensure `target_value` is respected. Can't have no disrespect.
//     #[test]
//     fn target_value_respected() {
//         let target_value = 1000_u64;

//         let candidates = (500..1500_u64)
//             .map(|value| WeightedValue {
//                 value,
//                 weight: 100,
//                 input_count: 1,
//                 is_segwit: false,
//             })
//             .collect::<super::Vec<_>>();

//         let opts = CoinSelectorOpt {
//             target_value: Some(target_value),
//             max_extra_target: 0,
//             target_feerate: 0.00,
//             long_term_feerate: None,
//             min_absolute_fee: 0,
//             base_weight: 10,
//             drain_weight: 10,
//             spend_drain_weight: 10,
//             min_drain_value: 10,
//         };

//         for (index, v) in candidates.iter().enumerate() {
//             let mut selector = CoinSelector::new(&candidates, &opts);
//             assert!(selector.select(index));

//             let res = selector.finish();
//             if v.value < opts.target_value.unwrap_or(0) {
//                 let err = res.expect_err("should have failed");
//                 assert_eq!(err.selected, v.value);
//                 assert_eq!(err.missing, target_value - v.value);
//                 assert_eq!(err.constraint, SelectionConstraint::MinAbsoluteFee);
//             } else {
//                 let sel = res.expect("should have succeeded");
//                 assert_eq!(sel.excess, v.value - opts.target_value.unwrap_or(0));
//             }
//         }
//     }

//     #[test]
//     fn drain_all() {
//         let candidates = (0..100)
//             .map(|_| WeightedValue {
//                 value: 666,
//                 weight: 166,
//                 input_count: 1,
//                 is_segwit: false,
//             })
//             .collect::<super::Vec<_>>();

//         let opts = CoinSelectorOpt {
//             target_value: None,
//             max_extra_target: 0,
//             target_feerate: 0.25,
//             long_term_feerate: None,
//             min_absolute_fee: 0,
//             base_weight: 10,
//             drain_weight: 100,
//             spend_drain_weight: 66,
//             min_drain_value: 1000,
//         };

//         let selection = CoinSelector::new(&candidates, &opts)
//             .select_until_finished()
//             .expect("should succeed");

//         assert!(selection.selected.len() > 1);
//         assert_eq!(selection.excess_strategies.len(), 1);

//         let (kind, strategy) = selection.best_strategy();
//         assert_eq!(*kind, ExcessStrategyKind::ToDrain);
//         assert!(strategy.recipient_value.is_none());
//         assert!(strategy.drain_value.is_some());
//     }

//     /// TODO: Tests to add:
//     /// * `finish` should ensure at least `target_value` is selected.
//     /// * actual feerate should be equal or higher than `target_feerate`.
//     /// * actual drain value should be equal or higher than `min_drain_value` (or else no drain).
//     fn _todo() {}
// }
