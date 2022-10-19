use super::*;
use crate::Vec;
use alloc::borrow::Cow;

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
pub struct Finished;
#[derive(Debug, Clone, Copy)]
pub struct Unfinished;

impl<'a> CoinSelector<'a> {}

#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub feerate: f32,
    pub min_fee: u64,
    pub value: u64,
}

impl Default for Target {
    fn default() -> Self {
        Self {
            feerate: 0.25,
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
        self.base_weight
            + self.selected_weight()
            + witness_header_extra_weight
            + vin_count_varint_extra_weight
    }

    pub fn weight_with_drain(&self, drain_weight: u32) -> u32 {
        self.current_weight() + drain_weight
    }

    pub fn excess(&self, target: Target) -> i64 {
        self.rate_excess(target.value, target.feerate)
            .min(self.abs_excess(target.value, target.min_fee))
    }

    /// The value that can be drained to a change output while maintaining the `feerate` and the `min_fee`
    pub fn drain_excess(&self, target: Target, drain_weight: u32) -> i64 {
        self.rate_excess_with_drain(target.value, target.feerate, drain_weight)
            .min(self.abs_excess(target.value, target.min_fee))
    }

    // /// Waste sum of all selected inputs.
    fn selected_waste(&self, feerate: f32, long_term_feerate: f32) -> f32 {
        self.selected_weight() as f32 * (feerate - long_term_feerate)
    }

    fn effective_value(&self, feerate: f32) -> i64 {
        self.selected_value() as i64 - self.fee_needed(feerate) as i64
    }

    fn effective_value_with_drain(&self, feerate: f32, drain_weight: u32) -> i64 {
        self.selected_value() as i64 - self.fee_needed_with_drain(feerate, drain_weight) as i64
    }

    fn rate_excess(&self, target_value: u64, feerate: f32) -> i64 {
        self.effective_value(feerate) - target_value as i64
    }

    fn rate_excess_with_drain(&self, target_value: u64, feerate: f32, drain_weight: u32) -> i64 {
        self.effective_value_with_drain(feerate, drain_weight) - target_value as i64
    }

    fn fee_needed(&self, feerate: f32) -> u64 {
        (self.current_weight() as f32 * feerate).ceil() as u64
    }

    fn fee_needed_with_drain(&self, feerate: f32, drain_weight: u32) -> u64 {
        (self.weight_with_drain(drain_weight) as f32 * feerate).ceil() as u64
    }

    fn abs_excess(&self, target_value: u64, abs_fee: u64) -> i64 {
        self.selected_value() as i64 - target_value as i64 - abs_fee as i64
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
        long_term_feerate: f32,
        drain_weights: Option<(u32, u32)>,
    ) -> f32 {
        let mut waste = self.selected_waste(target.feerate, long_term_feerate);
        match drain_weights {
            Some((drain_weight, drain_spend_weight)) => {
                waste += drain_weight as f32 * target.feerate
                    + drain_spend_weight as f32 * long_term_feerate;
            }
            None => waste += self.excess(target) as f32,
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
    pub fn select_until_target_met(&mut self, target: Target) -> Option<()> {
        self.select_until(|cs| cs.excess(target) >= 0)
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
        crate::coin_select::bnb::BnbIter::new(self, score_fn)
    }

    pub fn minimize_waste<'b, C>(
        &'b self,
        target: Target,
        long_term_feerate: f32,
        mut change_policy: C,
    ) -> impl Iterator<Item = Option<(CoinSelector<'a>, u32)>> + 'b
    where
        C: FnMut(&CoinSelector<'a>, Target) -> Option<(u32, u32)> + 'b,
    {
        let mut sorted_inputs = self.clone();
        sorted_inputs.sort_candidates_by_key(|(_, wv)| wv.effective_value(target.feerate));
        let rate_diff = target.feerate - long_term_feerate;

        let score_fn = move |cs: &CoinSelector<'a>, bound| {
            let drain_weights = change_policy(cs, target);

            if bound {
                let mut cs = cs.clone();

                let lower_bound = if rate_diff >= 0.0 {
                    // If feerate >= long_term_feerate then the least waste we can possibly have is the
                    // waste of what is currently selected + whatever we need to finish.
                    cs.select_until_target_met(target)?;
                    // NOTE: By passing the drain weights for current state we are implicitly
                    // assuming that if the change policy would add change now then it would if we
                    // add any more in the future. This assumption doesn't always hold but it helps
                    // a lot with branching as it demotes any selection after a change is added. It
                    // doesn't cause any harm in the case that rate_diff >= 0.0.
                    cs.waste(target, long_term_feerate, drain_weights).ceil() as u32
                } else {
                    // if the feerate < long_term_feerate then selecting everything remaining gives
                    // the lower bound on this selection's waste
                    cs.select_all();
                    if cs.excess(target) < 0 {
                        return None;
                    }
                    // NOTE the None here. If the long_term_feerate is low we actually don't want to
                    // assume we'll always add a change output if we have one now. We might
                    // add a low value input (decreases waste) which will remove the need for
                    // change.
                    cs.waste(target, long_term_feerate, None).ceil() as u32
                };

                Some(lower_bound)
            } else {
                let excess = cs.excess(target);
                if excess < 0 {
                    return None;
                }

                let score = cs.waste(target, long_term_feerate, drain_weights).ceil() as u32;
                Some(score)
            }
        };

        self.branch_and_bound(score_fn)
    }
}
