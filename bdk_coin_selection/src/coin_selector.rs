use super::*;
use crate::{bnb::BnBMetric, FeeRate};
use alloc::{borrow::Cow, collections::BTreeSet, vec::Vec};

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
    pub fn effective_value(&self, feerate: FeeRate) -> i64 {
        // We prefer undershooting the candidate's effective value (so we over estimate the fee of a
        // candidate). If we overshoot the candidate's effective value, it may be possible to find a
        // solution which does not meet the target feerate.
        self.value as i64 - (self.weight as f32 * feerate.spwu()).ceil() as i64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Drain {
    pub weight: u32,
    pub value: u64,
    pub spend_weight: u32,
}

impl Drain {
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_none(&self) -> bool {
        self == &Drain::none()
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
    }
}

/// [`CoinSelector`] is responsible for selecting and deselecting from a set of canididates.
#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    base_weight: u32,
    candidates: &'a [WeightedValue],
    selected: Cow<'a, BTreeSet<usize>>,
    banned: Cow<'a, BTreeSet<usize>>,
    candidate_order: Cow<'a, Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub feerate: FeeRate,
    pub min_fee: u64,
    pub value: u64,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            feerate: FeeRate::default_min_relay_fee(),
            min_fee: 0, // TODO figure out what the actual network rule is for this
            value: 0,
        }
    }
}

impl<'a> CoinSelector<'a> {
    // TODO: constructor should be number of outputs and output weight instead so we can keep track
    pub fn new(candidates: &'a [WeightedValue], base_weight: u32) -> Self {
        Self {
            base_weight,
            candidates,
            selected: Cow::Owned(Default::default()),
            banned: Cow::Owned(Default::default()),
            candidate_order: Cow::Owned((0..candidates.len()).collect()),
        }
    }

    pub fn candidates(
        &self,
    ) -> impl DoubleEndedIterator<Item = (usize, WeightedValue)> + ExactSizeIterator + '_ {
        self.candidate_order
            .iter()
            .map(|i| (*i, self.candidates[*i]))
    }

    pub fn candidate(&self, index: usize) -> WeightedValue {
        self.candidates[index]
    }

    pub fn deselect(&mut self, index: usize) -> bool {
        self.selected.to_mut().remove(&index)
    }

    pub fn apply_selection<T>(&self, candidates: &'a [T]) -> impl Iterator<Item = &'a T> + '_ {
        self.selected.iter().map(|i| &candidates[*i])
    }

    pub fn select(&mut self, index: usize) -> bool {
        assert!(index < self.candidates.len());
        self.selected.to_mut().insert(index)
    }

    pub fn select_next(&mut self) -> bool {
        let next = self.unselected_indexes().next();
        if let Some(next) = next {
            self.select(next);
            true
        } else {
            false
        }
    }

    pub fn ban(&mut self, index: usize) {
        self.banned.to_mut().insert(index);
    }

    pub fn banned(&self) -> &BTreeSet<usize> {
        &self.banned
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected.contains(&index)
    }

    pub fn is_selection_possible(&self, target: Target) -> bool {
        let mut test = self.clone();
        test.select_all_effective(target.feerate);
        test.excess(target, Drain::none()) >= 0
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

    pub fn input_weight(&self) -> u32 {
        let witness_header_extra_weight = self
            .selected()
            .find(|(_, wv)| wv.is_segwit)
            .map(|_| 2)
            .unwrap_or(0);
        let vin_count_varint_extra_weight = {
            let input_count = self.selected().map(|(_, wv)| wv.input_count).sum::<usize>();
            (varint_size(input_count) - 1) * 4
        };

        self.selected_weight() + witness_header_extra_weight + vin_count_varint_extra_weight
    }

    /// Absolute value sum of all selected inputs.
    pub fn selected_value(&self) -> u64 {
        self.selected
            .iter()
            .map(|&index| self.candidates[index].value)
            .sum()
    }

    /// Current weight of template tx + selected inputs.
    pub fn weight(&self, drain_weight: u32) -> u32 {
        // TODO take into account whether drain tips over varint for number of outputs
        //
        // TODO: take into account the witness stack length for each input
        self.base_weight + self.input_weight() + drain_weight
    }

    /// How much the current selection overshoots the value needed to acheive `target`.
    ///
    /// In order for the resulting transaction to be valid this must be 0.
    pub fn excess(&self, target: Target, drain: Drain) -> i64 {
        self.selected_value() as i64
            - target.value as i64
            - drain.value as i64
            - self.implied_fee(target.feerate, target.min_fee, drain.weight) as i64
    }

    /// The feerate the transaction would have if we were to use this selection of inputs to acheive
    /// the
    pub fn implied_feerate(&self, target_value: u64, drain: Drain) -> FeeRate {
        let numerator = self.selected_value() as i64 - target_value as i64 - drain.value as i64;
        let denom = self.weight(drain.weight);
        FeeRate::from_sat_per_wu(numerator as f32 / denom as f32)
    }

    pub fn implied_fee(&self, feerate: FeeRate, min_fee: u64, drain_weight: u32) -> u64 {
        ((self.weight(drain_weight) as f32 * feerate.spwu()).ceil() as u64).max(min_fee)
    }

    /// The value of the current selected inputs minus the fee needed to pay for the selected inputs
    pub fn effective_value(&self, feerate: FeeRate) -> i64 {
        self.selected_value() as i64 - (self.input_weight() as f32 * feerate.spwu()).ceil() as i64
    }

    // /// Waste sum of all selected inputs.
    fn selected_waste(&self, feerate: FeeRate, long_term_feerate: FeeRate) -> f32 {
        self.selected_weight() as f32 * (feerate.spwu() - long_term_feerate.spwu())
    }

    pub fn sort_candidates_by<F>(&mut self, mut cmp: F)
    where
        F: FnMut((usize, WeightedValue), (usize, WeightedValue)) -> core::cmp::Ordering,
    {
        let order = self.candidate_order.to_mut();
        order.sort_by(|a, b| cmp((*a, self.candidates[*a]), (*b, self.candidates[*b])))
    }

    pub fn sort_candidates_by_key<F, K>(&mut self, mut key_fn: F)
    where
        F: FnMut((usize, WeightedValue)) -> K,
        K: Ord,
    {
        self.sort_candidates_by(|a, b| key_fn(a).cmp(&key_fn(b)))
    }

    pub fn sort_candidates_by_descending_effective_value(&mut self, feerate: FeeRate) {
        self.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.effective_value(feerate)))
    }

    pub fn waste(
        &self,
        target: Target,
        long_term_feerate: FeeRate,
        drain: Drain,
        excess_discount: f32,
    ) -> f32 {
        debug_assert!(excess_discount >= 0.0 && excess_discount <= 1.0);
        let mut waste = self.selected_waste(target.feerate, long_term_feerate);

        if drain.is_none() {
            waste += self.excess(target, drain) as f32 * excess_discount.max(0.0).min(1.0);
        } else {
            waste += drain.weight as f32 * target.feerate.spwu()
                + drain.spend_weight as f32 * long_term_feerate.spwu();
        }

        waste
    }

    pub fn selected(&self) -> impl ExactSizeIterator<Item = (usize, WeightedValue)> + '_ {
        self.selected
            .iter()
            .map(|&index| (index, self.candidates[index]))
    }

    pub fn unselected(&self) -> impl DoubleEndedIterator<Item = (usize, WeightedValue)> + '_ {
        self.unselected_indexes().map(|i| (i, self.candidates[i]))
    }

    pub fn selected_indexes(&self) -> &BTreeSet<usize> {
        &self.selected
    }

    pub fn unselected_indexes(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.candidate_order
            .iter()
            .filter(|index| !(self.selected.contains(index) || self.banned.contains(index)))
            .map(|index| *index)
    }

    pub fn is_exhausted(&self) -> bool {
        self.unselected_indexes().next().is_none()
    }

    pub fn is_target_met(&self, target: Target, drain: Drain) -> bool {
        self.excess(target, drain) >= 0
    }

    pub fn select_all(&mut self) {
        loop {
            if !self.select_next() {
                break;
            }
        }
    }

    pub fn select_all_effective(&mut self, feerate: FeeRate) {
        // TODO: remove collect here
        for i in self.unselected_indexes().collect::<Vec<_>>() {
            if self.candidates[i].effective_value(feerate) > 0 {
                self.select(i);
            }
        }
    }

    #[must_use]
    pub fn select_until_target_met(&mut self, target: Target, drain: Drain) -> Option<()> {
        self.select_until(|cs| cs.is_target_met(target, drain))
    }

    #[must_use]
    pub fn select_until(
        &mut self,
        mut predicate: impl FnMut(&CoinSelector<'a>) -> bool,
    ) -> Option<()> {
        loop {
            if predicate(&*self) {
                break Some(());
            }

            if !self.select_next() {
                break None;
            }
        }
    }

    pub fn select_while(
        &mut self,
        mut predicate: impl FnMut(&CoinSelector<'a>) -> bool,
        // TODO: Remove this in favor of being able to reverse sort candidate order
        reverse: bool,
    ) {
        loop {
            let next = if reverse {
                self.unselected_indexes().rev().next()
            } else {
                self.unselected_indexes().next()
            };
            if let Some(next) = next {
                self.select(next);
                if !predicate(&*self) {
                    self.deselect(next);
                    return;
                }
            } else {
                return;
            }
        }
    }

    pub fn branch_and_bound<M: BnBMetric>(
        &self,
        metric: M,
    ) -> impl Iterator<Item = Option<(CoinSelector<'a>, M::Score)>> {
        crate::bnb::BnbIter::new(self.clone(), metric)
    }
}
