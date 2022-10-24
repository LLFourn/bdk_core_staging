use crate::{bnb::BnBMetric, ord_float::Ordf32, CoinSelector, Drain, FeeRate, Target};

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
        let current_change = change_lower_bound(cs, self.target, &self.change_policy);
        // 0 excess represents the most optimistic excess for the purposes of producing a
        // lower bound.
        let excess_multiplyer = 0.0;

        if rate_diff >= 0.0 {
            let mut cs = cs.clone();
            // If feerate >= long_term_feerate, You *might* think that the waste lower bound here is
            // just the fewest number of inputs we need to meet the target but **no**. Consider if
            // there is 1 sat remaining to reach target. Should you add all the weight of the next
            // input for the waste calculation? NO. Our goal is to have the minimum weight to reach
            // the target so we should only add a tiny fraction of the weight of the next input.
            //
            // Step one: select everything up until the input that hits the target.
            let target_not_met = cs.select_while(
                |cs, _| !cs.is_target_met(self.target, current_change),
                false,
            );

            if target_not_met {
                return None;
            }

            let weight_fraction = {
                // Figure out how weight from the next input we'd need to reach the target given its
                // sats-per-weight-unit value.
                let remaining = cs.excess(self.target, current_change).abs();
                let (_, next_wv) = cs.unselected().next().unwrap();
                (remaining as f32 / next_wv.value as f32) * next_wv.weight as f32
            };
            let weight_lower_bound = cs.selected_weight() as f32 + weight_fraction;
            let mut waste = weight_lower_bound * rate_diff;
            waste += current_change.waste(self.target.feerate, self.long_term_feerate);

            Some(Ordf32(waste))
        } else {
            // When long_term_feerate > current feerate each input by itself has negative waste.
            // This doesn't mean that waste monotonically decreases as you add inputs because
            // somewhere along the line adding an input might cause the change policy to add a
            // change ouput which could increase waste.
            //
            // So we have to try two things and we which one is best to find the lower bound:
            // 1. try selecting everything regardless of change
            let with_change_waste = {
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
                    excess_multiplyer,
                )
            };

            // 2. select the highest weight solution with no change
            let no_change_waste = {
                let mut cs_ = cs.clone();

                cs_.select_while(
                    |_, (_, wv)| wv.effective_value(self.target.feerate).0 < 0.0,
                    true,
                );
                let change_never_found = cs_.select_while(
                    |cs, _| (self.change_policy)(&cs, self.target).is_none(),
                    true,
                );
                let no_change_waste = cs_.waste(
                    self.target,
                    self.long_term_feerate,
                    Drain::none(),
                    excess_multiplyer,
                );
                if change_never_found {
                    debug_assert!(cs_.is_exhausted());
                }
                no_change_waste
            };

            let lower_bound = with_change_waste.min(no_change_waste);

            Some(Ordf32(lower_bound))
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
            fn requires_ordering_by_descending_effective_value(&self) -> Option<FeeRate> {
                None$(.or(self.$b.requires_ordering_by_descending_effective_value()))*
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
        let long_term_feerate_diff = 3.630499;
        let change_weight = 1;
        let change_spend_weight = 41;
        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb(feerate + 0.0f32.max(long_term_feerate_diff));
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
        let long_term_feerate_diff = -3.8413858;
        let change_weight = 1;
        let change_spend_weight = 1;
        let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
        let long_term_feerate =
            FeeRate::from_sat_per_vb((0.0f32).max(feerate + long_term_feerate_diff));
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
        naive_select.sort_candidates_by_descending_effective_value(target.feerate);
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

    proptest! {
        #![proptest_config(ProptestConfig {
            timeout: 3_000,
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
            let long_term_feerate = FeeRate::from_sat_per_vb(0.0f32.max(feerate + long_term_feerate_diff));
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
                .take(300_000)
                .filter_map(|(i, sol)| Some((i, sol?)))
                .last();

           match best {
                Some((_i, (sol, _score))) => {

                    let mut cmp_benchmarks = vec![
                        {
                            let mut naive_select = cs.clone();
                            naive_select.sort_candidates_by_descending_effective_value(target.feerate);
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

    // #[test]
    // fn prop_changeless() {

    // }
}
