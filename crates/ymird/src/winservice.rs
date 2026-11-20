//! `ymird --windows-service`: the Windows service `ymird` (account `NT SERVICE\ymird`), registered
//! by the WiX installer or `ymirctl service install --system`. Stop, shutdown and preshutdown
//! from the service control manager shut the daemon down cleanly, engines included.

use std::ffi::OsString;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use windows_service::service::{ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult, ServiceStatusHandle};
use windows_service::{define_windows_service, service_dispatcher};

use crate::Args;

const NAME: &str = "ymird";
/// How long the SCM should wait for engines to stop before it gives up on us.
const STOP_WAIT: Duration = Duration::from_secs(60);

static ARGS: OnceLock<Args> = OnceLock::new();

define_windows_service!(ffi_service_main, service_main);

/// Hands the process to the service control manager; returns when the service has stopped.
pub fn run(args: Args) -> Result<()> {
    let _ = ARGS.set(args);
    service_dispatcher::start(NAME, ffi_service_main).context("starting the service dispatcher (is this running as the ymird service?)")
}

fn service_main(_: Vec<OsString>) {
    if let Err(e) = serve() {
        tracing::error!("ymird service: {e:#}");
    }
}

fn status(handle: &ServiceStatusHandle, state: ServiceState, exit: u32, wait: Duration) {
    let controls =
        if state == ServiceState::Running { ServiceControlAccept::STOP | ServiceControlAccept::PRESHUTDOWN } else { ServiceControlAccept::empty() };
    let _ = handle.set_service_status(ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: controls,
        exit_code: if exit == 0 { ServiceExitCode::Win32(0) } else { ServiceExitCode::ServiceSpecific(exit) },
        checkpoint: 0,
        wait_hint: wait,
        process_id: None,
    });
}

fn serve() -> Result<()> {
    let args = ARGS.get().cloned().context("service arguments missing")?;
    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
    let handle = service_control_handler::register(NAME, move |c| match c {
        ServiceControl::Stop | ServiceControl::Shutdown | ServiceControl::Preshutdown => {
            let _ = stop_tx.send(true);
            ServiceControlHandlerResult::NoError
        }
        ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
        _ => ServiceControlHandlerResult::NotImplemented,
    })
    .context("registering the service control handler")?;
    status(&handle, ServiceState::StartPending, 0, Duration::from_secs(30));
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            status(&handle, ServiceState::Stopped, 1, Duration::ZERO);
            return Err(e.into());
        }
    };
    let result = rt.block_on(async {
        let daemon = crate::start(&args).await?;
        status(&handle, ServiceState::Running, 0, Duration::ZERO);
        tracing::info!("running as the Windows service {NAME}");
        tokio::select! {
            _ = stop_rx.wait_for(|s| *s) => tracing::info!("service stop requested"),
            _ = daemon.exit_requested() => {}
        }
        status(&handle, ServiceState::StopPending, 0, STOP_WAIT);
        daemon.shutdown().await;
        anyhow::Ok(())
    });
    // A start error is reported as a service-specific exit code, so the recovery actions
    // (`sc failureflag ymird 1`) restart it.
    status(&handle, ServiceState::Stopped, u32::from(result.is_err()), Duration::ZERO);
    result
}
