use std::collections::HashSet;
use std::{net, thread};

use nakamoto::client::network::Services;
use nakamoto::client::traits::Handle;
use nakamoto::client::Handle as ClientHandle;
use nakamoto::client::{chan, Client, Config, Domain, Event, Network};
use nakamoto::net::poll;

use bdk_cli::anyhow::{self, anyhow};

use bdk_chain::bitcoin::{BlockHash, Script, Transaction};
use bdk_chain::{BlockId, TxHeight};

use crate::Domains;

type Reactor = poll::Reactor<net::TcpStream>;

// Things to handle:
// - Block disconnected before filter processed, what to do?
// - Rescan from 0

pub struct CbfClient {
    pub handle: ClientHandle<poll::reactor::Waker>,
    pub setup: Option<CbfSetup>,
}

#[derive(Debug, Clone)]
pub enum CbfEvent {
    BlockMatched(BlockId, Vec<(Transaction, TxHeight)>),
    BlockDisconnected(BlockId),
    ChainSynced(Option<BlockId>),
}

pub struct CbfSetup {
    pub events: chan::Receiver<Event>,
    pub processed_height: Option<BlockId>,
    pub blocks_matched: HashSet<BlockHash>,
    pub peer_height: Option<u32>,
}

impl CbfClient {
    pub fn new(network: Network, domains: Domains) -> anyhow::Result<Self> {
        let domains = match domains {
            Domains::IPv4 => vec![Domain::IPV4],
            Domains::IPv6 => vec![Domain::IPV6],
            Domains::Both => vec![Domain::IPV4, Domain::IPV6],
        };

        let config = Config {
            network,
            domains,
            ..Config::default()
        };

        let client = Client::<Reactor>::new()?;
        let handle = client.handle();

        // Run the client on a different thread, to not block the main thread.
        // Note that when we sync we rescan from the latest point of agreement,
        // so it's not a problem if the client is running in the background
        thread::spawn(|| client.run(config).unwrap());

        Ok(CbfClient {
            handle,
            setup: None,
        })
    }

    pub fn sync_setup(
        &mut self,
        scripts: impl Iterator<Item = Script>,
        processed_height: Option<BlockId>,
    ) -> anyhow::Result<()> {
        println!("Looking for peers...");
        self.handle.wait_for_peers(1, Services::default())?;
        println!("Connected to at least one peer");

        let events = self.handle.events();
        let blocks_matched = HashSet::new();
        let peer_height = None;

        self.setup = Some(CbfSetup {
            events,
            processed_height,
            blocks_matched,
            peer_height,
        });

        let processed_height = processed_height.map(|h| h.height as u64).unwrap_or(0);

        // TODO: maybe we should check our latest point of agreement?
        self.handle.rescan(processed_height.., scripts)?;
        println!("Rescanning chain from height {:?}", processed_height);
        Ok(())
    }

    pub fn next_event(&mut self) -> anyhow::Result<CbfEvent> {
        if self.setup.is_none() {
            return Err(anyhow!("Need to call sync_setup first".to_string()));
        }
        let mut setup = self.setup.as_mut().expect("We check above");
        loop {
            chan::select! {
                recv(setup.events) -> event => {
                    let event = event?;
                    match event {
                        Event::PeerNegotiated { height, .. } => {
                            println!("* peer negotiated with height {:?}", height);
                            let height = height as u32;
                            if setup.peer_height < Some(height) {
                                setup.peer_height = Some(height);
                            }
                            if setup.processed_height.map(|h| h.height) == setup.peer_height && setup.blocks_matched.is_empty() {
                                // TODO: improve this!
                                // It might be that both me and this peer
                                // are lagging behind
                                return Ok(CbfEvent::ChainSynced(setup.processed_height));
                            }
                        }
                        Event::PeerHeightUpdated { height, .. } => {
                            let height = height as u32;
                            if setup.peer_height < Some(height) {
                                setup.peer_height = Some(height);
                            }
                        }
                        Event::BlockDisconnected { height, hash, .. } => {
                            setup.blocks_matched.remove(&hash);
                            return Ok(CbfEvent::BlockDisconnected(BlockId { hash, height: height as u32 }));
                        }
                        Event::BlockMatched { height, hash, transactions, .. } => {
                            let txs = transactions.into_iter().map(|tx| (tx, TxHeight::Confirmed(height as u32))).collect();
                            setup.blocks_matched.remove(&hash);

                            return Ok(CbfEvent::BlockMatched(BlockId { hash, height: height as u32 }, txs));
                        }
                        Event::FilterProcessed { matched, height, block, .. } => {
                            setup.processed_height = Some(BlockId { height: height as u32, hash: block});
                            if matched {
                                println!("* filter matched @ height {} : {:?}", height, &event);
                                setup.blocks_matched.insert(block);
                            }

                            if setup.processed_height.map(|h| h.height) == setup.peer_height && setup.blocks_matched.is_empty() {
                                return Ok(CbfEvent::ChainSynced(setup.processed_height));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

impl bdk_cli::Broadcast for CbfClient {
    type Error = nakamoto::client::handle::Error;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error> {
        println!("Looking for peers...");
        self.handle.wait_for_peers(1, Services::default())?;
        println!("Connected to at least one peer");

        self.handle.submit_transaction(tx.clone())?;
        Ok(())
    }
}
