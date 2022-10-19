use super::{CoinSelector, Target};
use crate::FeeRate;

pub fn minimize_waste<'a, C>(
    cs: &CoinSelector<'a>,
    bound: bool,
    target: Target,
    long_term_feerate: FeeRate,
    change_policy: C,
) -> Option<u32>
where
    C: Fn(&CoinSelector<'a>, Target) -> Option<(u32, u32)>,
{
    _minimize_waste(
        cs,
        bound,
        target,
        long_term_feerate,
        change_policy,
        |_, _| 0,
    )
}

fn _minimize_waste<'a, C, M>(
    cs: &CoinSelector<'a>,
    bound: bool,
    target: Target,
    long_term_feerate: FeeRate,
    change_policy: C,
    modifier: M,
) -> Option<u32>
where
    C: Fn(&CoinSelector<'a>, Target) -> Option<(u32, u32)>,
    M: Fn(&CoinSelector<'a>, Option<(u32, u32)>) -> u32,
{
    let drain_weights = change_policy(cs, target);
    // we're setting the drain value to 0 since we're not deciding that here
    let drain_value_and_weight = drain_weights.map(|dw| (0, dw.0));

    let rate_diff = target.feerate.spwu() - long_term_feerate.spwu();

    if bound {
        let mut cs = cs.clone();

        let lower_bound = if rate_diff >= 0.0 {
            // If feerate >= long_term_feerate then the least waste we can possibly have is the
            // waste of what is currently selected + whatever we need meet target.
            //
            // NOTE: we are passing in drain_value_and_weight here but it would be verry odd if it
            // was Some(_) and cs hadn't already met the target. It would mean that the change
            // policy dictates that drain must be included even when there is no excess value to
            // collect!
            cs.select_until_target_met(target, drain_value_and_weight)?;
            // NOTE: By passing the drain weights for current state we are implicitly assuming that
            // if the change policy would add change now then it would if we add any more inputs in
            // the future. This assumption doesn't always hold but it helps a lot with branching as
            // it demotes any selection after a change is added. It doesn't cause any harm in the
            // case that rate_diff >= 0.0.
            (cs.waste(target, long_term_feerate, drain_weights).ceil() as u32)
                .saturating_add(modifier(&cs, drain_weights))
        } else {
            // if the feerate < long_term_feerate then selecting everything remaining gives
            // the lower bound on this selection's waste
            cs.select_all();
            // NOTE the None for drainvalue and weight. If the long_term_feerate is low we actually don't
            // want to assume we'll always add a change output if we have one now. We might add a
            // low value input (decreases waste) which will remove the need for change because of
            // the extra fee it will require.
            if cs.excess(target, None) < 0 {
                return None;
            }
            (cs.waste(target, long_term_feerate, None).ceil() as u32)
                .saturating_add(modifier(&cs, drain_weights))
        };

        Some(lower_bound)
    } else {
        let excess = cs.excess(target, drain_value_and_weight);
        if excess < 0 {
            return None;
        }

        let score = cs.waste(target, long_term_feerate, drain_weights).ceil() as u32;
        Some(score)
    }
}

pub fn minimize_waste_sensible<'a, C>(
    cs: &CoinSelector<'a>,
    bound: bool,
    target: Target,
    long_term_feerate: FeeRate,
    change_policy: C,
) -> Option<u32>
where
    C: Fn(&CoinSelector<'a>, Target) -> Option<(u32, u32)>,
{
    let modifier = move |cs: &CoinSelector<'a>, drain_weights: Option<(u32, u32)>| {
        let drain_weight = drain_weights.map(|dw| dw.0);
        cs.implied_fee(target.feerate, target.min_fee, drain_weight) as u32
    };
    _minimize_waste(
        cs,
        bound,
        target,
        long_term_feerate,
        change_policy,
        modifier,
    )
}
