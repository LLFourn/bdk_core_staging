use super::{CoinSelector, Finished, Unfinished};
use crate::{collections::BinaryHeap, Vec};

#[derive(Debug)]
struct Branch<'a, O, H> {
    heuristic_score: O,
    depth: usize,
    selector: CoinSelector<'a, Unfinished>,
    already_scored: bool,
    hint: H,
}

impl<'a, O: Ord, H> Ord for Branch<'a, O, H> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // NOTE: reverse the order because we want a min-heap
        other.heuristic_score.cmp(&self.heuristic_score)
    }
}

impl<'a, O: PartialOrd, H> PartialOrd for Branch<'a, O, H> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        other.heuristic_score.partial_cmp(&self.heuristic_score)
    }
}

impl<'a, O: PartialEq, H> PartialEq for Branch<'a, O, H> {
    fn eq(&self, other: &Self) -> bool {
        self.heuristic_score == other.heuristic_score
    }
}

impl<'a, O: PartialEq, H> Eq for Branch<'a, O, H> {}

#[derive(Debug)]
pub(crate) struct BnbIter<'a, O, F, G, H> {
    queue: BinaryHeap<Branch<'a, O, H>>,
    pool: Vec<usize>,
    best: Option<O>,
    score_fn: F,
    heuristic_fn: G,
}

impl<'a, O, F, G, H> Iterator for BnbIter<'a, O, F, G, H>
where
    O: Ord + core::fmt::Debug + Clone,
    F: FnMut(&CoinSelector<'a, Finished>, H) -> Option<O>,
    G: FnMut(&CoinSelector<'a, Unfinished>, &[usize]) -> Option<(O, H)>,
{
    type Item = Option<(CoinSelector<'a, Finished>, O)>;

    fn next(&mut self) -> Option<Self::Item> {
        let branch = loop {
            let branch = self.queue.pop()?;
            self.insert_new_branches(&branch.selector, branch.depth);
            if !branch.already_scored {
                break branch;
            }
        };

        let finished_selector = match branch.selector.finish() {
            Some(finished_selector) => finished_selector,
            None => return Some(None),
        };

        let score = match (self.score_fn)(&finished_selector, branch.hint) {
            Some(score) => score,
            None => return Some(None),
        };

        match &self.best {
            Some(best_score) if score >= *best_score => Some(None),
            _ => {
                self.best = Some(score.clone());
                return Some(Some((finished_selector, score)));
            }
        }
    }
}

impl<'a, O, F, G, H> BnbIter<'a, O, F, G, H>
where
    G: FnMut(&CoinSelector<'a, Unfinished>, &[usize]) -> Option<(O, H)>,
    O: Ord,
{
    pub fn new<S>(selector: &CoinSelector<'a, S>, score_fn: F, heuristic_fn: G) -> Self {
        let pool = selector.unselected_indexes().collect();
        let selector = selector.clone().unfinish();

        let mut iter = BnbIter {
            queue: Default::default(),
            pool,
            best: None,
            score_fn,
            heuristic_fn,
        };

        iter.insert_new_branches(&selector, 0);

        iter
    }

    fn insert_new_branches(&mut self, cs: &CoinSelector<'a, Unfinished>, cur_depth: usize) {
        let remaining = &self.pool[cur_depth..];

        if remaining.is_empty() {
            return;
        }

        let mut inclusion_cs = cs.clone();
        inclusion_cs.select(self.pool[cur_depth]);
        let exclusion_cs = cs;

        for (child_cs, already_scored) in [(&inclusion_cs, false), (exclusion_cs, true)] {
            if let Some((heuristic, hint)) = (self.heuristic_fn)(child_cs, remaining) {
                if self.best.is_none() || self.best.as_ref().unwrap() > &heuristic {
                    self.queue.push(Branch {
                        heuristic_score: heuristic,
                        depth: cur_depth + 1,
                        selector: child_cs.clone(),
                        already_scored,
                        hint,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod test {

    use rand::{Rng, RngCore};

    use super::*;
    use crate::coin_select::{CoinSelector, CoinSelectorOpt, WeightedValue};

    fn test_wv(mut rng: impl RngCore) -> impl Iterator<Item = WeightedValue> {
        core::iter::repeat_with(move || {
            let value = rng.gen_range(0..100_000);
            let weight = rng.gen_range(0..1_000);
            WeightedValue {
                value,
                weight,
                input_count: rng.gen_range(1..2),
                is_segwit: rng.gen_bool(0.5),
            }
        })
    }

    #[test]
    fn todo_turn_into_proptest() {
        let mut wv = test_wv(rand::thread_rng());
        let num_canidates = 100;
        let solution: Vec<WeightedValue> = (0..10).map(|_| wv.next().unwrap()).collect();
        let target = solution.iter().map(|c| c.value).sum();
        let solution_length = solution.len();

        let mut candidates = solution;
        candidates.extend(wv.take(num_canidates - candidates.len()));
        candidates.sort_unstable_by_key(|wv| core::cmp::Reverse(wv.value));

        let cs = CoinSelector::new(
            &candidates,
            CoinSelectorOpt {
                target_value: target,
                base_weight: 0,
            },
        );

        let solutions = cs.branch_and_bound(
            |cs, _| Some((cs.abs_excess(0), cs.selected_weight())),
            |cs, candidates| {
                Some((
                    (
                        cs.abs_excess(0).max(0),
                        cs.iter_finished(candidates).next()?.selected_weight(),
                    ),
                    (),
                ))
            },
        );

        let (_i, (best, _score)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("it found a solution");
        dbg!(_i);

        dbg!(solution_length, best.selected().len());
        assert!(best.selected().len() <= solution_length);
        assert_eq!(best.selected_value(), target);
    }
}
