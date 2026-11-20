//! `ymir-engine` built in the test kit (the real one is a `ymird` binary, which tests can't name).

fn main() -> std::process::ExitCode {
    ymir_core::shim::main(std::env::args().skip(1).collect())
}
