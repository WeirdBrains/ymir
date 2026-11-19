use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = ymirctl::Cli::parse();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    if let Err(e) = ymirctl::run(cli, &mut std::io::stdout(), &mut input).await {
        eprintln!("ymirctl: {e:#}");
        std::process::exit(1);
    }
}
