use crate::{bnb::BnBMetric, CoinSelector, Drain, FeeRate, Target};

pub struct Waste<'c, C> {
    target: Target,
    long_term_feerate: FeeRate,
    change_policy: &'c C,
}

impl<'c, C> BnBMetric for Waste<'c, C>
where
    for<'a, 'b> C: Fn(&'b CoinSelector<'a>, Target) -> Drain,
{
    type Score = i32;

    fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        let drain = (self.change_policy)(cs, self.target);
        let excess = cs.excess(self.target, drain);
        if excess < 0 {
            return None;
        }
        let score = cs
            .waste(self.target, self.long_term_feerate, drain, 1.0)
            .round() as i32;
        Some(score)
    }
    fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        let rate_diff = self.target.feerate.spwu() - self.long_term_feerate.spwu();
        let current_change = change_lower_bound(cs, self.target, &self.change_policy);
        // 0 excess waste represents the most optimisitc lower bound
        let excess_discount = 0.0;

        if rate_diff >= 0.0 {
            let mut cs = cs.clone();
            // If feerate >= long_term_feerate then the least waste we can possibly have is the
            // waste of what is currently selected + whatever we need meet target.
            cs.select_until_target_met(self.target, current_change)?;
            let lower_bound = cs.waste(
                self.target,
                self.long_term_feerate,
                current_change,
                excess_discount,
            );
            Some(lower_bound.round() as i32)
        } else {
            let mut lower_bound = None;
            // When long_term_feerate > current feerate each input by itself has negative waste.
            // This doesn't mean that waste monotonically decreases as you add inputs because
            // somewhere along the line adding an input might cause the change policy to add a
            // change ouput which could increase waste.
            //
            // So we have to try two things and we which one is best to find the lower bound:
            //
            // // 1. select everything
            {
                let mut cs = cs.clone();
                cs.select_all();
                let change = (self.change_policy)(&cs, self.target);
                if cs.is_target_met(self.target, change) {
                    lower_bound =
                        Some(cs.waste(self.target, self.long_term_feerate, change, excess_discount))
                }
            }

            // 2. select as much as possible without adding change (only try if we don't have change right now).
            if current_change.is_none() {
                let mut cs = cs.clone();
                // select the lowest effective value candidates to minimize excess but maximise weight
                cs.select_while(|cs| (self.change_policy)(cs, self.target).is_none(), true);

                if cs.is_target_met(self.target, Drain::none()) {
                    let changeless_lower_bound = cs.waste(
                        self.target,
                        self.long_term_feerate,
                        Drain::none(),
                        excess_discount,
                    );
                    lower_bound = Some(lower_bound.unwrap_or(f32::MAX).min(changeless_lower_bound))
                }
            }
            lower_bound.map(|lb| lb.round() as i32)
        }
    }

    fn requires_ordering_by_descending_effective_value(&self) -> Option<FeeRate> {
        Some(self.target.feerate)
    }
}

pub struct Changeless<'c, C> {
    target: Target,
    change_policy: &'c C,
}

impl<'c, C> BnBMetric for Changeless<'c, C>
where
    for<'a, 'b> C: Fn(&'b CoinSelector<'a>, Target) -> Drain,
{
    type Score = bool;

    fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        let drain = (self.change_policy)(cs, self.target);
        if cs.excess(self.target, drain) > 0 {
            let has_drain = !drain.is_none();
            Some(has_drain)
        } else {
            None
        }
    }

    fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        Some(change_lower_bound(cs, self.target, &self.change_policy).is_some())
    }

    fn requires_ordering_by_descending_effective_value(&self) -> Option<FeeRate> {
        Some(self.target.feerate)
    }
}

// Returns a drain if the current selection and every possible future selection would have a change
// output (otherwise Drain::none()) by using the heurisitic that if it has change with the current
// selection and it has one when we select every negative effective value candidate then it will
// always have a drain. We are essentially assuming that the change_policy is monotone with respect
// to the excess of the selection.
//
// NOTE: this should stay private because it requires cs to be sorted by descending effective value
fn change_lower_bound<'a>(
    cs: &CoinSelector<'a>,
    target: Target,
    change_policy: &impl Fn(&CoinSelector<'a>, Target) -> Drain,
) -> Drain {
    let has_change_now = change_policy(cs, target).is_some();

    if has_change_now {
        let mut least_excess = cs.clone();
        cs.unselected()
            .rev()
            .take_while(|(_, wv)| wv.effective_value(target.feerate) < 0)
            .for_each(|(index, _)| {
                least_excess.select(index);
            });

        change_policy(&least_excess, target)
    } else {
        Drain::none()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::WeightedValue;
    use proptest::{
        prelude::*,
        test_runner::{RngAlgorithm, TestRng},
    };
    use rand::prelude::IteratorRandom;

    fn test_wv(mut rng: impl RngCore) -> impl Iterator<Item = WeightedValue> {
        core::iter::repeat_with(move || {
            let value = rng.gen_range(0..1_000);
            WeightedValue {
                value,
                weight: rng.gen_range(0..100),
                input_count: rng.gen_range(1..2),
                is_segwit: rng.gen_bool(0.5),
            }
        })
    }

    // this is probably a useful thing to have on CoinSelector but I don't want to design it yet
    fn randomly_satisfy_target_with_low_waste<'a>(
        cs: &CoinSelector<'a>,
        target: Target,
        long_term_feerate: FeeRate,
        change_policy: &impl Fn(&CoinSelector, Target) -> Drain,
        rng: &mut impl RngCore,
    ) -> Option<CoinSelector<'a>> {
        let mut cs = cs.clone();

        let mut last_waste: Option<f32> = None;
        while let Some(next) = cs.unselected_indexes().choose(rng) {
            cs.select(next);
            let change = change_policy(&cs, target);
            if cs.is_target_met(target, change) {
                let curr_waste = cs.waste(target, long_term_feerate, change, 1.0);
                if let Some(last_waste) = last_waste {
                    if curr_waste > last_waste {
                        break;
                    }
                }
                last_waste = Some(curr_waste);
            }
        }

        if cs.is_target_met(target, change_policy(&cs, target)) {
            Some(cs)
        } else {
            None
        }
    }

    proptest! {
        // #![proptest_config(ProptestConfig::with_cases(20))]
        #[test]
        fn prop_waste(
            num_inputs in 0usize..50,
            target in 0u64..25_000,
            feerate in 1.0f32..5.0,
            min_fee in 0u64..1_000,
            base_weight in 0u32..500,
            long_term_feerate in 1.0f32..5.0,
            change_weight in 1u32..100,
            change_spend_weight in 1u32..100,
        ) {
            let start = std::time::Instant::now();
            let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
            let feerate = FeeRate::from_sat_per_vb(feerate);
            let long_term_feerate = FeeRate::from_sat_per_vb(long_term_feerate);
            let drain = Drain {
                weight: change_weight,
                spend_weight: change_spend_weight,
                value: 0
            };

            let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
            let wv = test_wv(&mut rng);
            let candidates = wv.take(num_inputs).collect::<Vec<_>>();

            let cs = CoinSelector::new(&candidates, base_weight);

            let target = Target {
                value: target,
                feerate,
                min_fee
            };

            let solutions = cs.branch_and_bound(Waste {
                target,
                long_term_feerate,
                change_policy: &change_policy
            });


            let best = solutions
                .enumerate()
                .take(10_000)
                .inspect(|(i, _)| if start.elapsed().as_secs() > 1 {
                    // this vaguely means something is wrong and we should check it out
                    panic!("longer than a second elapsed on iteration {}", i);
                })
                .filter_map(|(i, sol)| Some((i, sol?)))
                .last();

           match best {
                Some((_i, (sol, _score))) => {

                    let mut cmp_benchmarks = vec![
                        {
                            let mut naive_select = cs.clone();
                            naive_select.sort_candidates_by_descending_effective_value(target.feerate);
                            naive_select.select_until_target_met(target, drain).expect("should be able to reach target");
                            naive_select
                        },
                        {
                            let mut all_selected = cs.clone();
                            all_selected.select_all();
                            all_selected
                        },
                    ];

                    cmp_benchmarks.extend((0..5).filter_map(|_|randomly_satisfy_target_with_low_waste(&cs, target, long_term_feerate, &change_policy, &mut rng)));

                    let sol_waste = sol.waste(target, long_term_feerate, change_policy(&sol, target), 1.0);

                    for (_bench_id, bench) in cmp_benchmarks.iter().enumerate() {
                        let bench_waste = bench.waste(target, long_term_feerate, change_policy(&bench, target), 1.0);
                        prop_assert!(sol_waste.round() <= bench_waste.round());
                    }
                },
                None => prop_assert!(!cs.is_selection_possible(target))
            }

            dbg!(start.elapsed());
        }
    }
}
