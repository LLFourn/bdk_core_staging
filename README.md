# BDK core staging

This is a repo for building out bdk_core before it can be integrated into bdk.

The plan is to have each kind of "blockchain" and each kind of "datastore" as its own crate. Everything will depend on bdk_core most likely but bdk_core only depends on `rust-bitcoin` and `rust-miniscript`.

In addition there is the `bdk_core_example` crate which has a basic command line wallet for seeing how it works.

See the [blog post] for an intro to the idea of `bdk_core`

## Things that have been built/started

###  `DescriptorTracker`

The [blog post] has more detail on the motivation behind this.

This decouples completely how you fetch data from how you store it.

#### TODOs

- I think we should keep a hash of all txids found out to a certain block. This help decide whether we are in sync with persistent storage for example and helps find where two sets of data diverge. This would be better than the current approach.
- The algorithm for tracking which txouts have been spent is not guaranteed to be correct if things happen in odd orders or you get conflicting things in the mempool. I think we will need an index of txo spends e.g. outpoints -> tx that spends them.

### `CoinSelector`

This keeps track of the value of the coins you have selected so far and whether the coin selection constraints have been satisfied. Improvements over bdk:

1. You can have **both** a feerate and absolute fee constraint.
2. The coin selection "algorithm" logic does not need to keep track of whether feerate has been satisfied yet etc. All this logic is done for you. (at least in coin selection algorithms that are greedy I don't know how well this idea will work with branch and bound).
3. No traits needed to be implemented to do coin selection. This is good because you can use bespoke application data like utxo labels etc without having to pass them into something implementing `CoinSelectionAlgorithm`.
4. `CoinSelector` tries checks if it complete at any stage both with and without change. In bdk the choice of change [is done after](https://github.com/bitcoindevkit/bdk/issues/147) coin selection which is sub-optimal.

#### TODOs

- implement branch and bound to see how that works
- Port bdk's `FeeRate`

## Big Things that are missing

### rpc and compact block filters

Incorporating them will probably change the API a bit.

### Persistent Storage

The approach here will start with a flat file db, get that working and then add the others that already exist.


### Choosing how you are going to spend an input

In bdk we use `max_satisfaction_weight` to figure out the weight of an input for coin selection. But couldn't we ask the application to choose how it would satisfy each input? Or maybe just find the way of satisfying the input with the least weight? This would allow us to use the weight for that witness in coin selection and also let us know what to set the csv/locktime to etc.

Without this too much has to be handled by the user. The `policy` module of bdk has some of the logic we need to some extent.

## Try it out

An example cli wallet is in `bdk_example`.
It does a full sync each time you run it and then runs your command.
By default it uses signet. You set these environment variables for your descriptors. These values already have coins and history.

This is not well tested. If something goes wrong it's probably a bug!

```
cd bdk_core_example
export DESCRIPTOR="tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)" CHANGE_DESCRIPTOR="tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)"

cargo run -- address list
cargo run -- address next
cargo run -- send 10000 <the new address> 
```

## Contribute

- open issues with respect to any aspect of the project
- make PRs



[blog post]: https://bitcoindevkit.org/blog/bdk-core-pt1/
