use crate::{
    bnb::BnBMetric, ord_float::Ordf32, CoinSelector, Drain, FeeRate, Target, WeightedValue,
};

pub struct Waste<'c, C> {
    target: Target,
    long_term_feerate: FeeRate,
    change_policy: &'c C,
}

impl<'c, C> BnBMetric for Waste<'c, C>
where
    for<'a, 'b> C: Fn(&'b CoinSelector<'a>, Target) -> Drain,
{
    type Score = Ordf32;

    fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        let drain = (self.change_policy)(cs, self.target);
        if !cs.is_target_met(self.target, drain) {
            return None;
        }
        let score = cs.waste(self.target, self.long_term_feerate, drain, 1.0);
        Some(Ordf32(score))
    }

    fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        let rate_diff = self.target.feerate.spwu() - self.long_term_feerate.spwu();
        // whether from this coin selection it's possible to avoid change
        let change_lower_bound = change_lower_bound(&cs, self.target, &self.change_policy);
        const IGNORE_EXCESS: f32 = 0.0;
        const INCLUDE_EXCESS: f32 = 1.0;

        if rate_diff >= 0.0 {
            // Our lower bound algorithms differ depending on whether we have already met the target or not.
            if cs.is_target_met(self.target, change_lower_bound) {
                let current_change = (self.change_policy)(&cs, self.target);

                // first lower bound candidate is just the selection itself
                let mut lower_bound = cs.waste(
                    self.target,
                    self.long_term_feerate,
                    current_change,
                    INCLUDE_EXCESS,
                );

                // but don't stop there we might be able to select more stuff to remove the change output!
                let should_explore_changeless =
                    // there might be the possibility of changeless
                    change_lower_bound.is_none()
                    // AND we currently have change
                    && current_change.is_some();

                if should_explore_changeless {
                    let selection_with_as_much_negative_ev_as_possible = cs
                        .clone()
                        .select_iter()
                        .rev()
                        .take_while(|(cs, _, wv)| {
                            wv.effective_value(self.target.feerate).0 < 0.0
                                && cs.is_target_met(self.target, Drain::none())
                        })
                        .last();

                    if let Some((cs, _, _)) = selection_with_as_much_negative_ev_as_possible {
                        let can_do_better_by_slurping =
                            cs.unselected().rev().next().and_then(|(_, wv)| {
                                if wv.effective_value(self.target.feerate).0 < 0.0 {
                                    Some(wv)
                                } else {
                                    None
                                }
                            });
                        let lower_bound_without_change = match can_do_better_by_slurping {
                            Some(finishing_input) => {
                                // NOTE we are slurping negative value here to try and reduce excess in
                                // the hopes of getting rid of the change output
                                let value_to_slurp = -cs.rate_excess(self.target, Drain::none());
                                let weight_to_extinguish_excess =
                                    slurp_wv(finishing_input, value_to_slurp, self.target.feerate);
                                let waste_to_extinguish_excess =
                                    weight_to_extinguish_excess * rate_diff;
                                let waste_after_excess_reduction = cs.waste(
                                    self.target,
                                    self.long_term_feerate,
                                    Drain::none(),
                                    IGNORE_EXCESS,
                                ) + waste_to_extinguish_excess;
                                waste_after_excess_reduction
                            }
                            None => cs.waste(
                                self.target,
                                self.long_term_feerate,
                                Drain::none(),
                                INCLUDE_EXCESS,
                            ),
                        };

                        lower_bound = lower_bound.min(lower_bound_without_change);
                    }
                }

                Some(Ordf32(lower_bound))
            } else {
                // If feerate >= long_term_feerate, You *might* think that the waste lower bound
                // here is just the fewest number of inputs we need to meet the target but **no**.
                // Consider if there is 1 sat remaining to reach target. Should you add all the
                // weight of the next input for the waste calculation? *No* this leaads to a
                // pesimistic lower bound even if we ignore the excess because it adds too much
                // weight.
                //
                // Step 1: select everything up until the input that hits the target.
                let (mut cs, slurp_index, to_slurp) = cs
                    .clone()
                    .select_iter()
                    .find(|(cs, _, _)| cs.is_target_met(self.target, change_lower_bound))?;

                cs.deselect(slurp_index);

                // Step 2: We pretend that the final input exactly cancels out the remaining excess
                // by taking whatever value we want from it but at the value per weight of the real
                // input.
                let ideal_next_weight = {
                    // satisfying absolute and feerate requires different calculations sowe do them
                    // both indepdently and find which requires the most weight of the next input.
                    let remaining_rate = cs.rate_excess(self.target, change_lower_bound);
                    let remaining_abs = cs.absolute_excess(self.target, change_lower_bound);

                    let weight_to_satisfy_abs =
                        remaining_abs.min(0) as f32 / to_slurp.value_pwu().0;
                    let weight_to_satisfy_rate =
                        slurp_wv(to_slurp, remaining_rate.min(0), self.target.feerate);
                    let weight_to_satisfy = weight_to_satisfy_abs.max(weight_to_satisfy_rate);
                    debug_assert!(weight_to_satisfy <= to_slurp.weight as f32);
                    weight_to_satisfy
                };
                let weight_lower_bound = cs.selected_weight() as f32 + ideal_next_weight;
                let mut waste = weight_lower_bound * rate_diff;
                waste += change_lower_bound.waste(self.target.feerate, self.long_term_feerate);

                Some(Ordf32(waste))
            }
        } else {
            // When long_term_feerate > current feerate each input by itself has negative waste.
            // This doesn't mean that waste monotonically decreases as you add inputs because
            // somewhere along the line adding an input might cause the change policy to add a
            // change ouput which could increase waste.
            //
            // So we have to try two things and we which one is best to find the lower bound:
            // 1. try selecting everything regardless of change
            let mut lower_bound = {
                let mut cs = cs.clone();
                // ... but first check that by selecting all effective we can actually reach target
                cs.select_all_effective(self.target.feerate);
                if !cs.is_target_met(self.target, Drain::none()) {
                    return None;
                }
                let change_at_value_optimum = (self.change_policy)(&cs, self.target);
                cs.select_all();
                // NOTE: we use the change from our "all effective" selection for min waste since
                // selecting all might not have change but in that case we'll catch it below.
                cs.waste(
                    self.target,
                    self.long_term_feerate,
                    change_at_value_optimum,
                    IGNORE_EXCESS,
                )
            };

            let look_for_changeless_solution = change_lower_bound.is_none();

            if look_for_changeless_solution {
                // 2. select the highest weight solution with no change
                let highest_weight_selection_without_change = cs
                    .clone()
                    .select_iter()
                    .rev()
                    .take_while(|(cs, _, wv)| {
                        wv.effective_value(self.target.feerate).0 < 0.0
                            || (self.change_policy)(&cs, self.target).is_none()
                    })
                    .last();

                if let Some((cs, _, _)) = highest_weight_selection_without_change {
                    let no_change_waste = cs.waste(
                        self.target,
                        self.long_term_feerate,
                        Drain::none(),
                        IGNORE_EXCESS,
                    );

                    lower_bound = lower_bound.min(no_change_waste)
                }
            }

            Some(Ordf32(lower_bound))
        }
    }

    fn requires_ordering_by_descending_value_pwu(&self) -> bool {
        true
    }
}

/// Used to pretend that a candidate had precisely `value_to_slurp` + fee needed to include it. It
/// tells you how much weight such a perfect candidate would have if it had the same value per
/// weight unit as `candidate`. This is useful for estimating a lower weight bound for a perfect
/// match.
fn slurp_wv(candidate: WeightedValue, value_to_slurp: i64, feerate: FeeRate) -> f32 {
    // the value per weight unit this candidate offers at feerate
    let value_per_wu = (candidate.value as f32 / candidate.weight as f32) - feerate.spwu();
    // return how much weight we need
    let weight_needed = value_to_slurp as f32 / value_per_wu;
    debug_assert!(weight_needed <= candidate.weight as f32);
    weight_needed.min(0.0)
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
        if cs.is_target_met(self.target, drain) {
            let has_drain = !drain.is_none();
            Some(has_drain)
        } else {
            None
        }
    }

    fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
        Some(change_lower_bound(cs, self.target, &self.change_policy).is_some())
    }

    fn requires_ordering_by_descending_value_pwu(&self) -> bool {
        true
    }
}

// Returns a drain if the current selection and every possible future selection would have a change
// output (otherwise Drain::none()) by using the heurisitic that if it has change with the current
// selection and it has one when we select every negative effective value candidate then it will
// always have a drain. We are essentially assuming that the change_policy is monotone with respect
// to the excess of the selection.
//
// NOTE: this should stay private because it requires cs to be sorted such that all negative
// effective value candidates are next to each other.
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
            .take_while(|(_, wv)| wv.effective_value(target.feerate) < Ordf32(0.0))
            .for_each(|(index, _)| {
                least_excess.select(index);
            });

        change_policy(&least_excess, target)
    } else {
        Drain::none()
    }
}

macro_rules! impl_for_tuple {
    ($($a:ident $b:tt)*) => {
        impl<$($a),*> BnBMetric for ($($a),*)
            where $($a: BnBMetric),*
        {
            type Score=($(<$a>::Score),*);

            #[allow(unused)]
            fn score<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
                Some(($(self.$b.score(cs)?),*))
            }
            #[allow(unused)]
            fn bound<'a>(&mut self, cs: &CoinSelector<'a>) -> Option<Self::Score> {
                Some(($(self.$b.bound(cs)?),*))
            }
            #[allow(unused)]
            fn requires_ordering_by_descending_value_pwu(&self) -> bool {
                [$(self.$b.requires_ordering_by_descending_value_pwu()),*].iter().all(|x| *x)

            }
        }
    };
}

impl_for_tuple!();
impl_for_tuple!(A 0 B 1);
impl_for_tuple!(A 0 B 1 C 2);
impl_for_tuple!(A 0 B 1 C 2 D 3);
impl_for_tuple!(A 0 B 1 C 2 D 3 E 4);

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
    #[allow(unused)]
    fn randomly_satisfy_target_with_low_waste<'a>(
        cs: &CoinSelector<'a>,
        target: Target,
        long_term_feerate: FeeRate,
        change_policy: &impl Fn(&CoinSelector, Target) -> Drain,
        rng: &mut impl RngCore,
    ) -> CoinSelector<'a> {
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
        cs
    }

    #[test]
    fn all_selected_except_one_is_optimal_and_awkward() {
        let num_inputs = 40;
        let target = 15578;
        let feerate = 8.190512;
        let min_fee = 0;
        let base_weight = 453;
        let long_term_feerate_diff = -3.630499;
        let change_weight = 1;
        let change_spend_weight = 41;
        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
        let feerate = FeeRate::from_sat_per_vb(feerate);
        let drain = Drain {
            weight: change_weight,
            spend_weight: change_spend_weight,
            value: 0,
        };

        let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
        let wv = test_wv(&mut rng);
        let candidates = wv.take(num_inputs).collect::<Vec<_>>();

        let cs = CoinSelector::new(&candidates, base_weight);
        let target = Target {
            value: target,
            feerate,
            min_fee,
        };

        let solutions = cs.branch_and_bound(Waste {
            target,
            long_term_feerate,
            change_policy: &change_policy,
        });

        let (_i, (best, score)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("it should have found solution");

        let mut all_selected = cs.clone();
        all_selected.select_all();
        let target_waste = all_selected.waste(
            target,
            long_term_feerate,
            change_policy(&all_selected, target),
            1.0,
        );
        assert_eq!(best.selected().len(), 39);
        assert!(score.0 < target_waste);
    }

    #[test]
    fn naive_effective_value_shouldnt_be_better() {
        let num_inputs = 23;
        let target = 1475;
        let feerate = 1.0;
        let min_fee = 989;
        let base_weight = 0;
        let long_term_feerate_diff = 3.8413858;
        let change_weight = 1;
        let change_spend_weight = 1;
        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
        let feerate = FeeRate::from_sat_per_vb(feerate);
        let drain = Drain {
            weight: change_weight,
            spend_weight: change_spend_weight,
            value: 0,
        };

        let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
        let wv = test_wv(&mut rng);
        let candidates = wv.take(num_inputs).collect::<Vec<_>>();

        let cs = CoinSelector::new(&candidates, base_weight);

        let target = Target {
            value: target,
            feerate,
            min_fee,
        };

        let solutions = cs.branch_and_bound(Waste {
            target,
            long_term_feerate,
            change_policy: &change_policy,
        });

        let (_i, (_best, score)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("should find solution");

        let mut naive_select = cs.clone();
        naive_select.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.value_pwu()));
        // we filter out failing onces below
        let _ = naive_select.select_until_target_met(target, drain);

        let bench_waste = naive_select.waste(
            target,
            long_term_feerate,
            change_policy(&naive_select, target),
            1.0,
        );

        assert!(score < Ordf32(bench_waste));
    }

    #[test]
    fn doesnt_take_too_long_to_finish() {
        let start = std::time::Instant::now();
        let num_inputs = 22;
        let target = 0;
        let feerate = 4.9522414;
        let min_fee = 0;
        let base_weight = 2;
        let long_term_feerate_diff = -0.17994404;
        let change_weight = 1;
        let change_spend_weight = 34;

        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
        let feerate = FeeRate::from_sat_per_vb(feerate);
        let drain = Drain {
            weight: change_weight,
            spend_weight: change_spend_weight,
            value: 0,
        };

        let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
        let wv = test_wv(&mut rng);
        let candidates = wv.take(num_inputs).collect::<Vec<_>>();

        let cs = CoinSelector::new(&candidates, base_weight);

        let target = Target {
            value: target,
            feerate,
            min_fee,
        };

        let solutions = cs.branch_and_bound(Waste {
            target,
            long_term_feerate,
            change_policy: &change_policy,
        });

        let (_i, (best, score)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("should find solution");

        if start.elapsed().as_millis() > 1_000 {
            dbg!(score, _i, change_policy(&best, target));
            println!("{}", best);
            panic!("took too long to finish");
        }
    }

    /// When long term feerate is lower than current adding new inputs should in general make things
    /// worse except in the case that we can get rid of the change output with negative effective
    /// value inputs. In this case the right answer to select everything.
    #[test]
    fn lower_long_term_feerate_but_still_need_to_select_all() {
        let num_inputs = 16;
        let target = 5586;
        let feerate = 9.397041;
        let min_fee = 0;
        let base_weight = 91;
        let long_term_feerate_diff = 0.22074795;
        let change_weight = 1;
        let change_spend_weight = 27;

        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb(0.0f32.max(feerate - long_term_feerate_diff));
        let feerate = FeeRate::from_sat_per_vb(feerate);
        let drain = Drain {
            weight: change_weight,
            spend_weight: change_spend_weight,
            value: 0,
        };

        let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
        let wv = test_wv(&mut rng);
        let candidates = wv.take(num_inputs).collect::<Vec<_>>();

        let cs = CoinSelector::new(&candidates, base_weight);

        let target = Target {
            value: target,
            feerate,
            min_fee,
        };

        let solutions = cs.branch_and_bound(Waste {
            target,
            long_term_feerate,
            change_policy: &change_policy,
        });
        let bench = {
            let mut all_selected = cs.clone();
            all_selected.select_all();
            all_selected
        };

        let (_i, (_sol, waste)) = solutions
            .enumerate()
            .filter_map(|(i, sol)| Some((i, sol?)))
            .last()
            .expect("should find solution");

        let bench_waste = bench.waste(
            target,
            long_term_feerate,
            change_policy(&bench, target),
            1.0,
        );

        assert!(waste <= Ordf32(bench_waste));
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            timeout: 3_000,
            cases: 1_000,
            ..Default::default()
        })]
        #[test]
        #[cfg(not(debug_assertions))] // too slow if compiling for debug
        fn prop_waste(
            num_inputs in 0usize..50,
            target in 0u64..25_000,
            feerate in 1.0f32..10.0,
            min_fee in 0u64..1_000,
            base_weight in 0u32..500,
            long_term_feerate_diff in -5.0f32..5.0,
            change_weight in 1u32..100,
            change_spend_weight in 1u32..100,
        ) {
            println!("=======================================");
            let start = std::time::Instant::now();
            let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
            let long_term_feerate = FeeRate::from_sat_per_vb(0.0f32.max(feerate - long_term_feerate_diff));
            let feerate = FeeRate::from_sat_per_vb(feerate);
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
                .filter_map(|(i, sol)| Some((i, sol?)))
                .last();

           match best {
                Some((_i, (sol, _score))) => {

                    let mut cmp_benchmarks = vec![
                        {
                            let mut naive_select = cs.clone();
                            naive_select.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.effective_value(target.feerate)));
                            // we filter out failing onces below
                            let _ = naive_select.select_until_target_met(target, drain);
                            naive_select
                        },
                        {
                            let mut all_selected = cs.clone();
                            all_selected.select_all();
                            all_selected
                        },
                        {
                            let mut all_effective_selected = cs.clone();
                            all_effective_selected.select_all_effective(target.feerate);
                            all_effective_selected
                        }
                    ];

                    // add some random selections -- technically it's possible that one of these is better but it's very unlikely if our algorithm is working correctly.
                    cmp_benchmarks.extend((0..10).map(|_|randomly_satisfy_target_with_low_waste(&cs, target, long_term_feerate, &change_policy, &mut rng)));

                    let cmp_benchmarks = cmp_benchmarks.into_iter().filter(|cs| cs.is_target_met(target, change_policy(&cs, target)));
                    let sol_waste = sol.waste(target, long_term_feerate, change_policy(&sol, target), 1.0);

                    for (_bench_id, bench) in cmp_benchmarks.enumerate() {
                        let bench_waste = bench.waste(target, long_term_feerate, change_policy(&bench, target), 1.0);
                        dbg!(_bench_id);
                        prop_assert!(sol_waste <= bench_waste);
                    }
                },
                None => {
                    dbg!(feerate - long_term_feerate);
                    prop_assert!(!cs.is_selection_plausible_with_change_policy(target, &change_policy));
                }
            }

            dbg!(start.elapsed());
        }
    }
}
