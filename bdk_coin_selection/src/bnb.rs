use crate::FeeRate;

use super::CoinSelector;
use alloc::collections::BinaryHeap;

#[derive(Debug)]
pub(crate) struct BnbIter<'a, M: BnBMetric> {
    queue: BinaryHeap<Branch<'a, M::Score>>,
    best: Option<M::Score>,
    /// The `BnBMetric` that will score each selection
    metric: M,
}

impl<'a, M: BnBMetric> Iterator for BnbIter<'a, M> {
    type Item = Option<(CoinSelector<'a>, M::Score)>;

    fn next(&mut self) -> Option<Self::Item> {
        let branch = self.queue.pop()?;
        if let Some(best) = &self.best {
            // If the next thing in queue is worse than our best we're done
            if *best < branch.lower_bound {
                return None;
            }
        }

        let selector = branch.selector;

        self.insert_new_branches(&selector);

        if branch.is_exclusion {
            return Some(None);
        }

        let score = match self.metric.score(&selector) {
            Some(score) => score,
            None => return Some(None),
        };

        match &self.best {
            Some(best_score) if score >= *best_score => Some(None),
            _ => {
                self.best = Some(score.clone());
                return Some(Some((selector, score)));
            }
        }
    }
}

impl<'a, M: BnBMetric> BnbIter<'a, M> {
    pub fn new(mut selector: CoinSelector<'a>, metric: M) -> Self {
        let mut iter = BnbIter {
            queue: BinaryHeap::default(),
            best: None,
            metric,
        };

        if let Some(feerate) = iter
            .metric
            .requires_ordering_by_descending_effective_value()
        {
            selector.sort_candidates_by_descending_effective_value(feerate);
        }

        iter.consider_adding_to_queue(&selector, false);

        iter
    }

    fn consider_adding_to_queue(&mut self, cs: &CoinSelector<'a>, is_exclusion: bool) {
        if let Some(heuristic) = self.metric.bound(cs) {
            if self.best.is_none() || self.best.as_ref().unwrap() > &heuristic {
                self.queue.push(Branch {
                    lower_bound: heuristic,
                    selector: cs.clone(),
                    is_exclusion,
                });
            }
        }
    }

    fn insert_new_branches(&mut self, cs: &CoinSelector<'a>) {
        if cs.is_exhausted() {
            return;
        }

        let next_unselected = cs.unselected_indexes().next().unwrap();
        let mut inclusion_cs = cs.clone();
        inclusion_cs.select(next_unselected);
        let mut exclusion_cs = cs.clone();
        exclusion_cs.ban(next_unselected);

        for (child_cs, is_exclusion) in [(&inclusion_cs, false), (&exclusion_cs, true)] {
            self.consider_adding_to_queue(child_cs, is_exclusion)
        }
    }
}

#[derive(Debug)]
struct Branch<'a, O> {
    lower_bound: O,
    selector: CoinSelector<'a>,
    is_exclusion: bool,
}

impl<'a, O: Ord> Ord for Branch<'a, O> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // NOTE: reverse comparision because we want a min-heap NOTE: Tiebreak equal scores based on
        // whether it's exlusion or not (preferring inclusion). We do this because early in a BnB
        (&other.lower_bound, other.is_exclusion).cmp(&(&self.lower_bound, self.is_exclusion))
    }
}

impl<'a, O: Ord> PartialOrd for Branch<'a, O> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, O: PartialEq> PartialEq for Branch<'a, O> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound == other.lower_bound
    }
}

impl<'a, O: PartialEq> Eq for Branch<'a, O> {}

pub trait BnBMetric {
    type Score: Ord + Clone + core::fmt::Debug;

    fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score>;
    fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score>;
    fn requires_ordering_by_descending_effective_value(&self) -> Option<FeeRate> {
        None
    }
}

#[cfg(test)]
mod test {
    use crate::{CoinSelector, Drain, FeeRate, Target, WeightedValue};
    use alloc::vec::Vec;
    use proptest::{
        prelude::*,
        test_runner::{RngAlgorithm, TestRng},
    };
    use rand::{Rng, RngCore};

    use super::BnBMetric;

    fn test_wv(mut rng: impl RngCore) -> impl Iterator<Item = WeightedValue> {
        core::iter::repeat_with(move || {
            let value = rng.gen_range(0..1_000);
            WeightedValue {
                value,
                weight: 100,
                input_count: rng.gen_range(1..2),
                is_segwit: rng.gen_bool(0.5),
            }
        })
    }

    struct MinExcessThenWeight {
        target: Target,
    }

    impl BnBMetric for MinExcessThenWeight {
        type Score = (i64, u32);

        fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
            if cs.excess(self.target, Drain::none()) < 0 {
                None
            } else {
                Some((cs.excess(self.target, Drain::none()), cs.selected_weight()))
            }
        }

        fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
            let lower_bound_excess = cs.excess(self.target, Drain::none()).max(0);
            let lower_bound_weight = {
                let mut cs = cs.clone();
                cs.select_until_target_met(self.target, Drain::none())?;
                cs.selected_weight()
            };
            Some((lower_bound_excess, lower_bound_weight))
        }
    }

    #[test]
    /// Detect regressions/improvements by making sure it always finds the solution in the same
    /// number of iterations.
    fn finds_an_exact_solution_in_n_iter() {
        let solution_len = 8;
        let num_additional_canidates = 50;

        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let mut wv = test_wv(&mut rng);

        let solution: Vec<WeightedValue> = (0..solution_len).map(|_| wv.next().unwrap()).collect();
        let solution_weight = solution.iter().map(|sol| sol.weight).sum();
        let target = solution.iter().map(|c| c.value).sum();

        let mut candidates = solution.clone();
        candidates.extend(wv.take(num_additional_canidates));
        candidates.sort_unstable_by_key(|wv| core::cmp::Reverse(wv.value));

        let cs = CoinSelector::new(&candidates, 0);

        let target = Target {
            value: target,
            // we're trying to find an exact selection value so set fees to 0
            feerate: FeeRate::zero(),
            min_fee: 0,
        };

        let solutions = cs.branch_and_bound(MinExcessThenWeight { target });

        let (i, (best, _score)) = solutions
            .enumerate()
            .take(807)
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("it found a solution");

        assert_eq!(i, 806);

        assert!(best.selected_weight() <= solution_weight);
        assert_eq!(best.selected_value(), target.value);
    }

    #[test]
    fn finds_solution_if_possible_in_n_iter() {
        let num_inputs = 18;
        let target = 8_314;
        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let wv = test_wv(&mut rng);
        let candidates = wv.take(num_inputs).collect::<Vec<_>>();
        let cs = CoinSelector::new(&candidates, 0);

        let target = Target {
            value: target,
            feerate: FeeRate::default_min_relay_fee(),
            min_fee: 0,
        };

        let solutions = cs.branch_and_bound(MinExcessThenWeight { target });

        let (i, (sol, _score)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("found a solution");

        assert_eq!(i, 176);
        let excess = sol.excess(target, Drain::none());
        assert_eq!(excess, 8);
    }

    proptest! {

        #[test]
        fn always_finds_solution_if_possible(num_inputs in 1usize..50, target in 0u64..10_000) {
            let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
            let wv = test_wv(&mut rng);
            let candidates = wv.take(num_inputs).collect::<Vec<_>>();
            let cs = CoinSelector::new(&candidates, 0);

            let target = Target {
                value: target,
                feerate: FeeRate::zero(),
                min_fee: 0,
            };

            let solutions = cs.branch_and_bound(MinExcessThenWeight { target });

            match solutions.enumerate().filter_map(|(i, sol)| Some((i, sol?))).last() {
                Some((_i, (sol, _score))) => assert!(sol.selected_value() >= target.value),
                _ => prop_assert!(!cs.is_selection_possible(target, Drain::none())),
            }
        }

        #[test]
        fn always_finds_exact_solution_eventually(
            solution_len in 1usize..10,
            num_additional_canidates in 0usize..100,
            num_preselected in 0usize..10
        ) {
            let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
            let mut wv = test_wv(&mut rng);

            let solution: Vec<WeightedValue> = (0..solution_len).map(|_| wv.next().unwrap()).collect();
            let target = solution.iter().map(|c| c.value).sum();
            let solution_weight = solution.iter().map(|sol| sol.weight).sum();

            let mut candidates = solution.clone();
            candidates.extend(wv.take(num_additional_canidates));

            let mut cs = CoinSelector::new(&candidates, 0);
            for i in 0..num_preselected.min(solution_len) {
                cs.select(i);
            }

            // sort in descending value
            cs.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.value));

            let target = Target {
                value: target,
                // we're trying to find an exact selection value so set fees to 0
                feerate: FeeRate::zero(),
                min_fee: 0
            };

            let solutions = cs.branch_and_bound(MinExcessThenWeight { target });

            let (_i, (best, _score)) = solutions
                .enumerate()
                .filter_map(|(i, sol)| Some((i, sol?)))
                .last()
                .expect("it found a solution");



            prop_assert!(best.selected_weight() <= solution_weight);
            prop_assert_eq!(best.selected_value(), target.value);
        }
    }
}
