use crate::{CoinSelector, Drain, FeeRate, Target};

pub fn min_value(drain: Drain, min_value: u64) -> impl Fn(&CoinSelector, Target) -> Drain {
    debug_assert!(drain.is_some());
    move |cs, target| {
        if cs.excess(target, Drain::none()) >= drain.value as i64 {
            drain
        } else {
            Drain::none()
        }
    }
}

pub fn min_waste(
    drain: Drain,
    long_term_feerate: FeeRate,
) -> impl Fn(&CoinSelector, Target) -> Drain {
    debug_assert!(drain.is_some());
    move |cs, target| {
        if cs.excess(target, Drain::none())
            > (drain.spend_weight as f32 * long_term_feerate.spwu()).ceil() as i64
        {
            drain
        } else {
            Drain::none()
        }
    }
}
