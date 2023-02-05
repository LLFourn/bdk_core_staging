use alloc::vec::Vec;
use alloc::string::String;


use bdk_chain::{ FullTxOut, sparse_chain::ChainPosition, collections::{ HashSet, HashMap} };
use crate::WeightedValue;
use crate::TXIN_BASE_WEIGHT;

#[derive(Clone, Debug)]
pub enum CoinGroupingStrategy {
    AddressReuse,
}

pub type CoinGroup = (WeightedValue, HashSet<usize>);

impl From<&CoinGroup> for WeightedValue {
    fn from(group: &CoinGroup) -> Self {
        group.0
    }
}

#[derive(Debug, Clone)]
pub enum CoinGroupError {
    UnknownStrategy(String)
}


impl core::fmt::Display for CoinGroupError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoinGroupError::UnknownStrategy(strategy) => write!(f, "{} grouping strategy is not supported", strategy)
        }
    }
}

impl std::error::Error for CoinGroupError {}



impl core::str::FromStr for CoinGroupingStrategy {
    type Err = CoinGroupError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "address-reuse" => CoinGroupingStrategy::AddressReuse,
            unknown => return Err(CoinGroupError::UnknownStrategy(String::from(unknown)))
        })
    }
}


pub fn apply_grouping<P: ChainPosition, AK: bdk_tmp_plan::CanDerive + Clone>(candidates: &Vec<(bdk_tmp_plan::Plan<AK>, FullTxOut<P>)>, grouping_strategy: Option<CoinGroupingStrategy>) -> Vec<CoinGroup> {
    let mut script_coingroup_map = HashMap::new();
    match grouping_strategy {
        Some(strategy) => match strategy {
            CoinGroupingStrategy::AddressReuse => {
                candidates.iter()
                .enumerate()
                .for_each(|(idx, (plan, utxo))| {
                    let coin_group = script_coingroup_map.entry(utxo.txout.script_pubkey.clone()).or_insert((WeightedValue::new(
                        utxo.txout.value,
                        plan.expected_weight() as _,
                        plan.witness_version().is_some(),
                    ),
                    HashSet::from([idx])));
                    coin_group.1.insert(idx);
                    coin_group.0.value += utxo.txout.value;
                    coin_group.0.weight += plan.expected_weight() as u32 + TXIN_BASE_WEIGHT;
                    coin_group.0.input_count += 1;
                    coin_group.0.is_segwit |= plan.witness_version().is_some();
                });
                script_coingroup_map.into_values().collect::<Vec<CoinGroup>>()
            }
        },
        None => {
            candidates.iter()
            .enumerate()
            .map(|(idx, (plan, utxo))| {
                (WeightedValue::new(
                    utxo.txout.value,
                    plan.expected_weight() as u32,
                    plan.witness_version().is_some(),
                ),
                HashSet::from([idx]))
            })
            .collect::<Vec<CoinGroup>>()
        }
        
    }
}