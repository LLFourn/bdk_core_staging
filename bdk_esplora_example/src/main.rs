use bdk_chain::bitcoin::Network;
use bdk_esplora_example::esplora::Client;

const DEFAULT_PARALLEL_REQUESTS: u8 = 5;
use bdk_cli::{
    anyhow,
    clap::{self, Subcommand},
};

#[derive(Subcommand, Debug, Clone)]
enum EsploraCommands {
    /// Scans the addresses in the wallet using esplora API.
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: usize,
    },
    /// Scans particular addresses using esplora API
    Sync {
        /// Scan all the unused addresses
        #[clap(long)]
        unused: bool,
        /// Scan the script addresses that have unspent outputs
        #[clap(long)]
        unspent: bool,
        /// Scan every address that you have derived
        #[clap(long)]
        all: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let (args, keymap, keychain_tracker, db) = bdk_cli::init::<EsploraCommands, _>()?;
    let esplora_url = match args.network {
        Network::Bitcoin => "https://mempool.space/api",
        Network::Testnet => "https://mempool.space/testnet/api",
        Network::Regtest => "http://localhost:3002",
        Network::Signet => "https://mempool.space/signet/api",
    };

    let client = Client::new(esplora_url, DEFAULT_PARALLEL_REQUESTS)?;

    let esplora_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(esplora_cmd) => esplora_cmd,
        general_command => {
            return bdk_cli::handle_commands(
                general_command,
                client,
                &keychain_tracker,
                &db,
                args.network,
                &keymap,
            )
        }
    };

    match esplora_cmd {
        EsploraCommands::Scan { stop_gap } => {
            bdk_esplora_example::run_scan(stop_gap, &keychain_tracker, &db, &client)?;
        }
        EsploraCommands::Sync {
            unused,
            unspent,
            all,
        } => {
            bdk_esplora_example::run_sync(unused, unspent, all, &keychain_tracker, &db, &client)?;
        }
    }

    Ok(())
}
