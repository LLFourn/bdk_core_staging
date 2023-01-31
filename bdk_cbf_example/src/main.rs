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
    //dbg!(&keychain_tracker.txout_index);

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
            // TODO: I'm 99% sure this is not how I should handle the stop_gap,
            // but I don't have any better idea
            // In this way you might have to scan multiple times before being able
            // to see all your transactions: let's say you have 0 scripts used in
            // the database, but stop_gap + 1 scripts used on the blockchain: here
            // we derive until stop_gap, and you'll have to scan again before being
            // able to see the txs sent to the `stop_gap + 1`th script
            // keychain_tracker.txout_index.pad_all_with_unused(stop_gap);

            //dbg!("Keychain tracker before everything: {:?}", &keychain_tracker);
            // TODO: too many collects here!
            // let spk_iterators = keychain_tracker
            //     .txout_index
            //     .stored_scripts_of_all_keychains()
            //     .into_values()
            //     .flat_map(|s| s.map(|(_, s)| s.clone()).collect::<Vec<_>>())
            //     .collect::<Vec<_>>();
            let ch = client.sync(&mut keychain_tracker, stop_gap).unwrap();
            //dbg!("Keychain tracker after sync: {:?}", &keychain_tracker);
            //dbg!(&ch);
            db.append_changeset(&ch)?;
            keychain_tracker.apply_changeset(ch);
            //dbg!("Keychain tracker end: {:?}", &keychain_tracker);
            //dbg!(&keychain_tracker.txout_index);
            return bdk_cli::handle_commands(
                bdk_cli::Commands::<CbfCommands>::Balance,
                client,
                &mut keychain_tracker,
                &mut db,
                args.network,
                &keymap,
            );
        }
    }
}
