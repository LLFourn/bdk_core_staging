#![allow(dead_code)]
use bitcoin::{Script, Transaction};
pub use nakamoto_client;
use nakamoto_client::chan::RecvError;
use nakamoto_client::{
    chan::Receiver, client::Publisher, handle::Handle, protocol, Client, Config, Event,
    Handle as ClientHandle,
};
use nakamoto_common::network::Services;
use std::net::SocketAddr;
use std::{net::TcpStream, thread};

type Reactor = nakamoto_net_poll::Reactor<TcpStream, Publisher>;

/// A CBF Node Client wrapper
pub struct CbfNode {
    network: nakamoto_client::Network,
    scripts: Vec<Script>,
    receiver: Receiver<Event>,
    handle: ClientHandle<Reactor>,
}

impl CbfNode {
    pub fn new(network: bitcoin::Network, scripts: Vec<Script>) -> Self {
        let cbf_client = Client::<Reactor>::new().unwrap();
        let client_cfg = Config {
            listen: vec![], // Don't listen for incoming connections.
            protocol: protocol::Config {
                network: network.into(),
                ..protocol::Config::default()
            },
            ..Config::default()
        };

        let handle = cbf_client.handle();
        thread::spawn(|| {
            cbf_client.run(client_cfg).unwrap();
        });

        handle.wait_for_peers(1, Services::Chain).unwrap();

        let receiver = handle.subscribe();

        Self {
            network: network.into(),
            scripts,
            receiver,
            handle,
        }
    }

    pub fn scan(&self, from: u32) {
        let _ = self
            .handle
            .rescan((from as u64).., self.scripts.iter().cloned());
    }

    pub fn add_addresses(&mut self, addresses: Vec<Script>) {
        self.scripts.extend(addresses);
        let _ = self.handle.watch(self.scripts.clone().into_iter());
    }

    pub fn add_peers(&self, peers: Vec<SocketAddr>) {
        for peer in peers {
            self.handle.connect(peer).unwrap();
        }
    }

    pub fn diconnect(&self, peer: SocketAddr) {
        self.handle.disconnect(peer).unwrap();
    }

    pub fn braodcast(&self, tx: Transaction) {
        self.handle.submit_transaction(tx).unwrap();
    }

    pub fn get_next_event(&self) -> Result<Event, RecvError> {
        self.receiver.recv()
    }

    // Destroys self
    pub fn shutdown(self) {
        self.handle.shutdown().unwrap();
    }
}
