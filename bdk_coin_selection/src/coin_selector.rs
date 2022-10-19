use super::*;
use crate::FeeRate;
use alloc::borrow::Cow;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

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

/// [`CoinSelector`] is responsible for selecting and deselecting from a set of canididates.
#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    base_weight: u32,
    candidates: &'a [WeightedValue],
    selected: Cow<'a, BTreeSet<usize>>,
    banned: Cow<'a, BTreeSet<usize>>,
    candidate_order: Cow<'a, Vec<usize>>,
}

impl<'a> CoinSelector<'a> {}

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
    pub fn weight(&self, drain_weight: Option<u32>) -> u32 {
        // TODO take into account whether drain tips over varint for number of outputs
        let drain_weight = drain_weight.unwrap_or(0);
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
        self.base_weight
            + self.selected_weight()
            + witness_header_extra_weight
            + vin_count_varint_extra_weight
            + drain_weight
    }

    /// How much the current selection overshoots the value needed to acheive `target`.
    ///
    /// You may optionally pass in a drain `(value, weight)` tuple. A useful use of this is to pass
    /// in `0` for the value (but use the correct weight) which make the function return precisely
    /// the amonut of satoshis to set the drain output to so you have 0 excess (which is usually your
    /// goal).
    pub fn excess(&self, target: Target, drain: Option<(u64, u32)>) -> i64 {
        let (drain_value, drain_weight) = drain.unwrap_or((0, 0));
        let rate_excess = self.effective_value(target.feerate, Some(drain_weight))
            - drain_value as i64
            - target.value as i64;
        let abs_excess = self.selected_value() as i64
            - target.min_fee as i64
            - drain_value as i64
            - target.value as i64;
        rate_excess.min(abs_excess)
    }

    pub fn implied_feerate(&self, target_value: u64, drain: Option<(u64, u32)>) -> FeeRate {
        let (drain_value, drain_weight) = drain.unwrap_or((0, 0));
        let numerator = self.selected_value() as i64 - target_value as i64 - drain_value as i64;
        let denom = self.weight(Some(drain_weight));
        FeeRate::from_sats_per_wu(numerator as f32 / denom as f32)
    }

    pub fn implied_fee(&self, feerate: FeeRate, min_fee: u64, drain_weight: Option<u32>) -> u64 {
        ((self.weight(drain_weight) as f32 * feerate.spwu()).ceil() as u64).max(min_fee)
    }

    // /// Waste sum of all selected inputs.
    fn selected_waste(&self, feerate: FeeRate, long_term_feerate: FeeRate) -> f32 {
        self.selected_weight() as f32 * (feerate.spwu() - long_term_feerate.spwu())
    }

    fn effective_value(&self, feerate: FeeRate, drain_weight: Option<u32>) -> i64 {
        self.selected_value() as i64 - self.implied_fee(feerate, 0, drain_weight) as i64
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

    pub fn waste(
        &self,
        target: Target,
        long_term_feerate: FeeRate,
        drain_weights: Option<(u32, u32)>,
    ) -> f32 {
        let mut waste = self.selected_waste(target.feerate, long_term_feerate);
        match drain_weights {
            Some((drain_weight, drain_spend_weight)) => {
                waste += drain_weight as f32 * target.feerate.spwu()
                    + drain_spend_weight as f32 * long_term_feerate.spwu();
            }
            None => waste += self.excess(target, None) as f32,
        }

        waste
    }

    pub fn selected(&self) -> impl ExactSizeIterator<Item = (usize, &'a WeightedValue)> + '_ {
        self.selected
            .iter()
            .map(|&index| (index, &self.candidates[index]))
    }

    pub fn unselected(&self) -> impl Iterator<Item = (usize, &'a WeightedValue)> + '_ {
        self.unselected_indexes().map(|i| (i, &self.candidates[i]))
    }

    pub fn selected_indexes(&self) -> &BTreeSet<usize> {
        &self.selected
    }

    pub fn unselected_indexes(&self) -> impl Iterator<Item = usize> + '_ {
        self.candidate_order
            .iter()
            .filter(|index| !self.selected.contains(index) && !self.banned.contains(index))
            .map(|index| *index)
    }

    pub fn exhausted(&self) -> bool {
        self.unselected_indexes().next().is_none()
    }

    pub fn select_all(&mut self) {
        loop {
            if !self.select_next() {
                break;
            }
        }
    }

    #[must_use]
    pub fn select_until_target_met(
        &mut self,
        target: Target,
        drain_value_and_weight: Option<(u64, u32)>,
    ) -> Option<()> {
        self.select_until(|cs| cs.excess(target, drain_value_and_weight) >= 0)
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

    pub fn branch_and_bound<O, F>(
        &self,
        score_fn: F,
    ) -> impl Iterator<Item = Option<(CoinSelector<'a>, O)>>
    where
        O: Ord + core::fmt::Debug + Clone,
        F: FnMut(&CoinSelector<'a>, bool) -> Option<O>,
    {
        crate::bnb::BnbIter::new(self, score_fn)
    }
}
