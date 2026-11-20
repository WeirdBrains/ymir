//! `ymir-engine`: runs llama.cpp, MLX-LM or mistral.rs for one model, chosen by the model
//! source, behind `llama-server`'s interface (see `ymir_core::shim`). The supervisor starts it;
//! it is not meant to be run by hand.

fn main() -> std::process::ExitCode {
    ymir_core::shim::main(std::env::args().skip(1).collect())
}
