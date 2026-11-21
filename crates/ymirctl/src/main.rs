use clap::Parser;

#[tokio::main]
async fn main() {
    let cli = ymirctl::Cli::parse();
    // The MCP server owns stdin and stdout for the protocol: don't lock them here.
    let result = if matches!(cli.cmd, ymirctl::Cmd::Mcp(_)) {
        ymirctl::run(cli, &mut std::io::sink(), &mut std::io::empty()).await
    } else {
        let stdin = std::io::stdin();
        let mut input = stdin.lock();
        ymirctl::run(cli, &mut std::io::stdout(), &mut input).await
    };
    if let Err(e) = result {
        eprintln!("ymirctl: {e:#}");
        std::process::exit(1);
    }
}
