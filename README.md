# BDK core staging

This is a repo for building out bdk_core before it can be integrated into bdk.

## Components

The `bdk_core` project has three main components in order of importance:

### 1. `bdk_chain`

The goal of this component is give wallets the mechanisms needed to:

1. Figure out what data they need to fetch
2. Process that data in a way that never leads to inconsistent states
3. Fully index that data and expose it so that it can be consumed without friction.


Our design goals for these mechanisms are:

1. Data source agnostic -- nothing in `bdk_chain` cares about where you get data from or whether you
   do it synchronously or asynchronously. if you know a fact about the blockchain you can
   just tell `bdk_chain`'s APIs about it that information will be integrated if it can be done
   consistently.
2. Error free APIs
3. Data persistence agnostic. `bdk_chain` doesn't care where you cache on-chain data, what you cache or how you fetch it.

TODO:

- [x] Chain and data indexing
- [x] Persistant storage (see [file_store](./bdk_chain/src/file_store.rs))
- [x] Working esplora example (see [bdk_esplora_example](./bdk_esplora_example))
- [x] Working electrum example (see [bdk_electrum_example](./bdk_electrum_example))
- [ ] Working bitcoin core rpc block-by-block example (see [#89](https://github.com/LLFourn/bdk_core_staging/pull/89))
- [ ] Working bitcoin core rpc wallet sync example (see [#79](https://github.com/LLFourn/bdk_core_staging/pull/79))
- [ ] Working compact block filters example using nakamoto (see [#153](https://github.com/LLFourn/bdk_core_staging/pull/153)).
- [ ] Feerate calculation for RBF and CPFP.
- [ ] Complete transaction building module (Coin control, coin selection, satisfaction with planning module).

### 2. Miniscript planning module

This component is about properly using miniscript's potential to know exactly which outputs you can
spend and the most efficient way to spend them *before* trying to spend them. This allows you to
know precisely how much weight spending an output will add for coin selection. This is an important
feature in a taproot world since different spending paths can have vastly different weight.

The PR to miniscript has already been made: https://github.com/rust-bitcoin/rust-miniscript/pull/481

### 3. Coin selection, transaction building and signing

With the planning module, we'll be able to re-engineer coin selection and transaction building as well.


#### Coin selection

We've designed and implemented a robust coin selection API which allows users to choose what metric
they want to optimise for during a branch and bound search.

Coin selection PR: https://github.com/LLFourn/bdk_core_staging/pull/46


#### Transaction building

Transaction building issue: https://github.com/LLFourn/bdk_core_staging/issues/30

## Development and Release plan

The release dates for the first release we will try to make for the first releases are:

- 10-02-2023: Release `bdk_chain v0.1.0` -- working with esplora/electrum data
- 24-02-2023: Release `bdk v1.0.0-alpha.0`
- 10-03-2023: Release `bdk_chain v.2.0` -- working with blocks form CBF and bitcoin RPC
- 24-03-2023: Release `bdk v1.0.0-alpha.1`

From there we will continue to develop `bdk_chain` and the other components and as they are
integrated into `bdk` we will continue to make new alpha/beta releases of `bdk` until we have
something suitable for a release candidate. Before making a release candidate we'll need the
planning module to be merged into miniscript and released.

## Try it out

We have examples in their own crates like `bdk_esplora_example` and `bdk_electrum_example` with more coming soon.

To use them move into the directory and try running some commands.
First you'll need to set some descriptors.

```
export DESCRIPTOR="tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/0/*)" CHANGE_DESCRIPTOR="tr([73c5da0a/86'/0'/0']xprv9xgqHN7yz9MwCkxsBPN5qetuNdQSUttZNKw1dcYTV4mkaAFiBVGQziHs3NRSWMkCzvgjEe3n9xV8oYywvM8at9yRqyaZVz6TYYhX98VjsUk/1/*)"
```

### Initialize the data from the chain


``` sh
cargo run -- scan # for electrum and esplora
cargo run -- balance
```

### A plain BIP86 TR wallet

```
cargo run -- address list
cargo run -- address next
cargo run -- send 10000 <the new address>
```

### Script path spending works too

```
export DESCRIPTOR="tr(xpub6BgBgsespWvERF3LHQu6CnqdvfEvtMcQjYrcRzx53QJjSxarj2afYWcLteoGVky7D3UKDP9QyrLprQ3VCECoY49yfdDEHGCtMMj92pReUsQ/0/*,pk(xprv9s21ZrQH143K3ngkqk9y72BYSJTZ1ngfTFGFtxCwfP9pKqcMzn6aCP3mZoY8qMEqUjkxC2BkDUVLw77qbyGt66BbE7g3nt8JAGGkcTe4kWZ/0/*))"
```

## Contribute

- open issues with respect to any aspect of the project
- make PRs



[blog post]: https://bitcoindevkit.org/blog/bdk-core-pt1/
