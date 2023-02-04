use bdk_chain::TxHeight;
use bdk_cli::{
    anyhow,
    clap::{self, Args, Subcommand},
};

mod cbf;

#[derive(Args, Debug, Clone)]
struct CbfArgs {
    // TODO: disable ipv6 or ipv4
    // TODO: network
}

#[derive(Subcommand, Debug, Clone)]
enum CbfCommands {
    Scan {
        /// When a gap this large has been found for a keychain it will stop.
        #[clap(long, default_value = "5")]
        stop_gap: u32,
    },
}

fn main() -> anyhow::Result<()> {
    println!("Loading wallet from db...");
    let (args, keymap, mut keychain_tracker, mut db) =
        bdk_cli::init::<CbfArgs, CbfCommands, TxHeight>()?;
    println!("Wallet loaded.");

    let mut client = cbf::CbfClient::new(args.network.into())?;

    let cbf_cmd = match args.command {
        bdk_cli::Commands::ChainSpecific(cbf_cmd) => cbf_cmd,
        general_cmd => {
            return bdk_cli::handle_commands(
                general_cmd,
                client,
                &mut keychain_tracker,
                &mut db,
                args.network,
                &keymap,
            );
        }
    };

    match cbf_cmd {
        CbfCommands::Scan { stop_gap } => {
            let ch = client.sync(&mut keychain_tracker, stop_gap).unwrap();
            db.append_changeset(&ch)?;
            keychain_tracker.apply_changeset(ch);
            Ok(())
        }
    }
}
