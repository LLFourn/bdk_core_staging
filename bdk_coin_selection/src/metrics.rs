use crate::{bnb::BnBMetric, ord_float::Ordf32, CoinSelector, Drain, Target};
mod waste;
pub use waste::*;

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

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::WeightedValue;
//     use proptest::{
//         prelude::*,
//         test_runner::{RngAlgorithm, TestRng},
//     };
//     use rand::prelude::IteratorRandom;

//     fn test_wv(mut rng: impl RngCore) -> impl Iterator<Item = WeightedValue> {
//         core::iter::repeat_with(move || {
//             let value = rng.gen_range(0..1_000);
//             WeightedValue {
//                 value,
//                 weight: rng.gen_range(0..100),
//                 input_count: rng.gen_range(1..2),
//                 is_segwit: rng.gen_bool(0.5),
//             }
//         })
//     }

//     // this is probably a useful thing to have on CoinSelector but I don't want to design it yet
//     #[allow(unused)]
//     fn randomly_satisfy_target_with_low_waste<'a>(
//         cs: &CoinSelector<'a>,
//         target: Target,
//         long_term_feerate: FeeRate,
//         change_policy: &impl Fn(&CoinSelector, Target) -> Drain,
//         rng: &mut impl RngCore,
//     ) -> CoinSelector<'a> {
//         let mut cs = cs.clone();

//         let mut last_waste: Option<f32> = None;
//         while let Some(next) = cs.unselected_indexes().choose(rng) {
//             cs.select(next);
//             let change = change_policy(&cs, target);
//             if cs.is_target_met(target, change) {
//                 let curr_waste = cs.waste(target, long_term_feerate, change, 1.0);
//                 if let Some(last_waste) = last_waste {
//                     if curr_waste > last_waste {
//                         break;
//                     }
//                 }
//                 last_waste = Some(curr_waste);
//             }
//         }
//         cs
//     }

//     #[test]
//     fn all_selected_except_one_is_optimal_and_awkward() {
//         let num_inputs = 40;
//         let target = 15578;
//         let feerate = 8.190512;
//         let min_fee = 0;
//         let base_weight = 453;
//         let long_term_feerate_diff = -3.630499;
//         let change_weight = 1;
//         let change_spend_weight = 41;
//         let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
//         let long_term_feerate =
//             FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
//         let feerate = FeeRate::from_sat_per_vb(feerate);
//         let drain = Drain {
//             weight: change_weight,
//             spend_weight: change_spend_weight,
//             value: 0,
//         };

//         let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
//         let wv = test_wv(&mut rng);
//         let candidates = wv.take(num_inputs).collect::<Vec<_>>();

//         let cs = CoinSelector::new(&candidates, base_weight);
//         let target = Target {
//             value: target,
//             feerate,
//             min_fee,
//         };

//         let solutions = cs.branch_and_bound(Waste {
//             target,
//             long_term_feerate,
//             change_policy: &change_policy,
//         });

//         let (_i, (best, score)) = solutions
//             .enumerate()
//             .filter_map(|(i, sol)| Some((i, sol?)))
//             .last()
//             .expect("it should have found solution");

//         let mut all_selected = cs.clone();
//         all_selected.select_all();
//         let target_waste = all_selected.waste(
//             target,
//             long_term_feerate,
//             change_policy(&all_selected, target),
//             1.0,
//         );
//         assert_eq!(best.selected().len(), 39);
//         assert!(score.0 < target_waste);
//     }

//     #[test]
//     fn naive_effective_value_shouldnt_be_better() {
//         let num_inputs = 23;
//         let target = 1475;
//         let feerate = 1.0;
//         let min_fee = 989;
//         let base_weight = 0;
//         let long_term_feerate_diff = 3.8413858;
//         let change_weight = 1;
//         let change_spend_weight = 1;
//         let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
//         let long_term_feerate =
//             FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
//         let feerate = FeeRate::from_sat_per_vb(feerate);
//         let drain = Drain {
//             weight: change_weight,
//             spend_weight: change_spend_weight,
//             value: 0,
//         };

//         let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
//         let wv = test_wv(&mut rng);
//         let candidates = wv.take(num_inputs).collect::<Vec<_>>();

//         let cs = CoinSelector::new(&candidates, base_weight);

//         let target = Target {
//             value: target,
//             feerate,
//             min_fee,
//         };

//         let solutions = cs.branch_and_bound(Waste {
//             target,
//             long_term_feerate,
//             change_policy: &change_policy,
//         });

//         let (_i, (_best, score)) = solutions
//             .enumerate()
//             .filter_map(|(i, sol)| Some((i, sol?)))
//             .last()
//             .expect("should find solution");

//         let mut naive_select = cs.clone();
//         naive_select.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.value_pwu()));
//         // we filter out failing onces below
//         let _ = naive_select.select_until_target_met(target, drain);

//         let bench_waste = naive_select.waste(
//             target,
//             long_term_feerate,
//             change_policy(&naive_select, target),
//             1.0,
//         );

//         assert!(score < Ordf32(bench_waste));
//     }

//     #[test]
//     fn doesnt_take_too_long_to_finish() {
//         let start = std::time::Instant::now();
//         let num_inputs = 22;
//         let target = 0;
//         let feerate = 4.9522414;
//         let min_fee = 0;
//         let base_weight = 2;
//         let long_term_feerate_diff = -0.17994404;
//         let change_weight = 1;
//         let change_spend_weight = 34;

//         let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
//         let long_term_feerate =
//             FeeRate::from_sat_per_vb((0.0f32).max(feerate - long_term_feerate_diff));
//         let feerate = FeeRate::from_sat_per_vb(feerate);
//         let drain = Drain {
//             weight: change_weight,
//             spend_weight: change_spend_weight,
//             value: 0,
//         };

//         let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
//         let wv = test_wv(&mut rng);
//         let candidates = wv.take(num_inputs).collect::<Vec<_>>();

//         let cs = CoinSelector::new(&candidates, base_weight);

//         let target = Target {
//             value: target,
//             feerate,
//             min_fee,
//         };

//         let solutions = cs.branch_and_bound(Waste {
//             target,
//             long_term_feerate,
//             change_policy: &change_policy,
//         });

//         let (_i, (best, score)) = solutions
//             .enumerate()
//             .filter_map(|(i, sol)| Some((i, sol?)))
//             .last()
//             .expect("should find solution");

//         if start.elapsed().as_millis() > 1_000 {
//             dbg!(score, _i, change_policy(&best, target));
//             println!("{}", best);
//             panic!("took too long to finish");
//         }
//     }

//     /// When long term feerate is lower than current adding new inputs should in general make things
//     /// worse except in the case that we can get rid of the change output with negative effective
//     /// value inputs. In this case the right answer to select everything.
//     #[test]
//     fn lower_long_term_feerate_but_still_need_to_select_all() {
//         let num_inputs = 16;
//         let target = 5586;
//         let feerate = 9.397041;
//         let min_fee = 0;
//         let base_weight = 91;
//         let long_term_feerate_diff = 0.22074795;
//         let change_weight = 1;
//         let change_spend_weight = 27;

//         let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
//         let long_term_feerate =
//             FeeRate::from_sat_per_vb(0.0f32.max(feerate - long_term_feerate_diff));
//         let feerate = FeeRate::from_sat_per_vb(feerate);
//         let drain = Drain {
//             weight: change_weight,
//             spend_weight: change_spend_weight,
//             value: 0,
//         };

//         let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
//         let wv = test_wv(&mut rng);
//         let candidates = wv.take(num_inputs).collect::<Vec<_>>();

//         let cs = CoinSelector::new(&candidates, base_weight);

//         let target = Target {
//             value: target,
//             feerate,
//             min_fee,
//         };

//         let solutions = cs.branch_and_bound(Waste {
//             target,
//             long_term_feerate,
//             change_policy: &change_policy,
//         });
//         let bench = {
//             let mut all_selected = cs.clone();
//             all_selected.select_all();
//             all_selected
//         };

//         let (_i, (_sol, waste)) = solutions
//             .enumerate()
//             .filter_map(|(i, sol)| Some((i, sol?)))
//             .last()
//             .expect("should find solution");

//         let bench_waste = bench.waste(
//             target,
//             long_term_feerate,
//             change_policy(&bench, target),
//             1.0,
//         );

//         assert!(waste <= Ordf32(bench_waste));
//     }

//     proptest! {
//         #![proptest_config(ProptestConfig {
//             timeout: 3_000,
//             cases: 1_000,
//             ..Default::default()
//         })]
//         #[test]
//         #[cfg(not(debug_assertions))] // too slow if compiling for debug
//         fn prop_waste(
//             num_inputs in 0usize..50,
//             target in 0u64..25_000,
//             feerate in 1.0f32..10.0,
//             min_fee in 0u64..1_000,
//             base_weight in 0u32..500,
//             long_term_feerate_diff in -5.0f32..5.0,
//             change_weight in 1u32..100,
//             change_spend_weight in 1u32..100,
//         ) {
//             println!("=======================================");
//             let start = std::time::Instant::now();
//             let mut rng = TestRng::deterministic_rng(RngAlgorithm::ChaCha);
//             let long_term_feerate = FeeRate::from_sat_per_vb(0.0f32.max(feerate - long_term_feerate_diff));
//             let feerate = FeeRate::from_sat_per_vb(feerate);
//             let drain = Drain {
//                 weight: change_weight,
//                 spend_weight: change_spend_weight,
//                 value: 0
//             };

//             let change_policy = crate::change_policy::no_waste(drain, long_term_feerate);
//             let wv = test_wv(&mut rng);
//             let candidates = wv.take(num_inputs).collect::<Vec<_>>();

//             let cs = CoinSelector::new(&candidates, base_weight);

//             let target = Target {
//                 value: target,
//                 feerate,
//                 min_fee
//             };

//             let solutions = cs.branch_and_bound(Waste {
//                 target,
//                 long_term_feerate,
//                 change_policy: &change_policy
//             });

//             let best = solutions
//                 .enumerate()
//                 .filter_map(|(i, sol)| Some((i, sol?)))
//                 .last();

//            match best {
//                 Some((_i, (sol, _score))) => {

//                     let mut cmp_benchmarks = vec![
//                         {
//                             let mut naive_select = cs.clone();
//                             naive_select.sort_candidates_by_key(|(_, wv)| core::cmp::Reverse(wv.effective_value(target.feerate)));
//                             // we filter out failing onces below
//                             let _ = naive_select.select_until_target_met(target, drain);
//                             naive_select
//                         },
//                         {
//                             let mut all_selected = cs.clone();
//                             all_selected.select_all();
//                             all_selected
//                         },
//                         {
//                             let mut all_effective_selected = cs.clone();
//                             all_effective_selected.select_all_effective(target.feerate);
//                             all_effective_selected
//                         }
//                     ];

//                     // add some random selections -- technically it's possible that one of these is better but it's very unlikely if our algorithm is working correctly.
//                     cmp_benchmarks.extend((0..10).map(|_|randomly_satisfy_target_with_low_waste(&cs, target, long_term_feerate, &change_policy, &mut rng)));

//                     let cmp_benchmarks = cmp_benchmarks.into_iter().filter(|cs| cs.is_target_met(target, change_policy(&cs, target)));
//                     let sol_waste = sol.waste(target, long_term_feerate, change_policy(&sol, target), 1.0);

//                     for (_bench_id, bench) in cmp_benchmarks.enumerate() {
//                         let bench_waste = bench.waste(target, long_term_feerate, change_policy(&bench, target), 1.0);
//                         dbg!(_bench_id);
//                         prop_assert!(sol_waste <= bench_waste);
//                     }
//                 },
//                 None => {
//                     dbg!(feerate - long_term_feerate);
//                     prop_assert!(!cs.is_selection_plausible_with_change_policy(target, &change_policy));
//                 }
//             }

//             dbg!(start.elapsed());
//         }
//     }
// }
