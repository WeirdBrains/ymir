# Builds Ymir's Windows installers with the WiX Toolset 7 CLI:
#   make-installer.ps1 -ReleaseDir DIR -Version X.Y.Z -Arch x64|arm64 -OutDir dist
#       the MSI (ymir-X.Y.Z-ARCH.msi) from an unpacked release zip (ymird.exe at its top)
#   make-installer.ps1 -Setup -MsiX64 FILE [-MsiArm64 FILE] -Version X.Y.Z -OutDir dist
#       YmirSetup-X.Y.Z.exe, a bundle that installs the MSI for the machine's architecture
#
# Needs wix.exe and the WixToolset.UI, Util and BootstrapperApplications extensions of the same
# version in the global cache:
#   wix extension add -acceptEula wix7 -g WixToolset.UI.wixext/7.0.1 WixToolset.Util.wixext/7.0.1 WixToolset.BootstrapperApplications.wixext/7.0.1
# Every wix command passes `-acceptEula wix7`: WiX 7 binaries are used under the Open Source
# Maintenance Fee EULA v1.1 (https://docs.firegiant.com/wix/osmf/), which m3talsmith accepted for
# WeirdBrains projects on 2026-11-12. The installers are unsigned.
param(
  [string]$ReleaseDir,
  [Parameter(Mandatory=$true)][string]$Version,
  [ValidateSet("x64", "arm64")][string]$Arch = "x64",
  [Parameter(Mandatory=$true)][string]$OutDir,
  [switch]$Setup,
  [string]$MsiX64,
  [string]$MsiArm64,
  [string]$WixVersion = "7.0.1"
)
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
# MSI versions are numeric: 0.2.0-rc.1 installs as 0.2.0.
$msiVersion = ($Version -split "[-+]")[0]

$wix = Get-Command wix.exe -ErrorAction Stop
$have = (& $wix.Source --version).Trim()
if (-not $have.StartsWith("$WixVersion")) { Write-Error "wix.exe is $have, expected $WixVersion"; exit 1 }

# The License pane needs RTF; generate it from LICENSE so the two never drift.
$licenseText = [IO.File]::ReadAllText((Join-Path $repoRoot "LICENSE"))
$escaped = foreach ($line in ($licenseText -replace "`r", "") -split "`n") {
  $l = $line.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
  -join ($l.ToCharArray() | ForEach-Object { if ([int]$_ -gt 127) { "\u" + [int]$_ + "?" } else { [string]$_ } })
}
$licenseRtf = Join-Path ([IO.Path]::GetTempPath()) ("ymir-license-" + [guid]::NewGuid() + ".rtf")
[IO.File]::WriteAllText($licenseRtf, "{\rtf1\ansi\deff0{\fonttbl{\f0\fmodern Consolas;}}\f0\fs18 " + ($escaped -join "\par`r`n") + "}", [Text.Encoding]::ASCII)

function Invoke-MsiQuery([string]$Path, [string]$Sql) {
  $installer = New-Object -ComObject WindowsInstaller.Installer
  $db = $installer.GetType().InvokeMember("OpenDatabase", "InvokeMethod", $null, $installer, @($Path, 0))
  $view = $db.GetType().InvokeMember("OpenView", "InvokeMethod", $null, $db, @($Sql))
  $view.GetType().InvokeMember("Execute", "InvokeMethod", $null, $view, $null) | Out-Null
  $rows = @()
  while ($true) {
    $rec = $view.GetType().InvokeMember("Fetch", "InvokeMethod", $null, $view, $null)
    if ($null -eq $rec) { break }
    $rows += $rec.GetType().InvokeMember("StringData", "GetProperty", $null, $rec, @(1))
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($rec)
  }
  $view.GetType().InvokeMember("Close", "InvokeMethod", $null, $view, $null) | Out-Null
  foreach ($o in @($view, $db, $installer)) { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($o) }
  [GC]::Collect(); [GC]::WaitForPendingFinalizers()
  return ,$rows
}

try {
  if ($Setup) {
    if (!$MsiX64 -or !(Test-Path -LiteralPath $MsiX64)) { Write-Error "-Setup needs -MsiX64 FILE"; exit 1 }
    $exe = Join-Path $OutDir "YmirSetup-$Version.exe"
    $defs = @("-d", "ProductVersion=$msiVersion", "-d", "LicenseRtf=$licenseRtf", "-d", "MsiX64=$((Resolve-Path $MsiX64).Path)")
    if ($MsiArm64) { $defs += @("-d", "MsiArm64=$((Resolve-Path $MsiArm64).Path)") }
    & $wix.Source build -nologo -acceptEula wix7 -arch x64 -ext "WixToolset.BootstrapperApplications.wixext/$WixVersion" @defs `
      -pdbtype none -o $exe (Join-Path $PSScriptRoot "bundle.wxs")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (!(Test-Path -LiteralPath $exe)) { Write-Error "wix build reported success but $exe is missing"; exit 1 }
    Write-Host "Setup created at $exe"
    exit 0
  }

  if (!$ReleaseDir -or !(Test-Path -LiteralPath (Join-Path $ReleaseDir "ymird.exe"))) { Write-Error "-ReleaseDir must hold ymird.exe (an unpacked release zip)"; exit 1 }
  $ReleaseDir = (Resolve-Path $ReleaseDir).Path.TrimEnd('\')
  $msi = Join-Path $OutDir "ymir-$Version-$Arch.msi"
  & $wix.Source build -nologo -acceptEula wix7 -arch $Arch `
    -ext "WixToolset.UI.wixext/$WixVersion" -ext "WixToolset.Util.wixext/$WixVersion" `
    -d "ReleaseDir=$ReleaseDir" -d "ProductVersion=$msiVersion" -d "LicenseRtf=$licenseRtf" `
    -culture en-US -pdbtype none -o $msi (Join-Path $PSScriptRoot "ymir.wxs")
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  if (!(Test-Path -LiteralPath $msi)) { Write-Error "wix build reported success but $msi is missing"; exit 1 }
} finally {
  Remove-Item -LiteralPath $licenseRtf -Force -ErrorAction SilentlyContinue
}

# Check what was built.
$svcAccount = Invoke-MsiQuery $msi "SELECT ``StartName`` FROM ``ServiceInstall`` WHERE ``ServiceInstall``='svcInstall'"
if ($svcAccount.Count -ne 1 -or $svcAccount[0] -ne "NT SERVICE\ymird") { Write-Error "MSI check failed: service account ($svcAccount)"; exit 1 }
$svcArgs = Invoke-MsiQuery $msi "SELECT ``Arguments`` FROM ``ServiceInstall`` WHERE ``ServiceInstall``='svcInstall'"
if ($svcArgs.Count -ne 1 -or -not $svcArgs[0].StartsWith('--windows-service --home "[CommonAppDataFolder]Ymir"')) { Write-Error "MSI check failed: service arguments ($svcArgs)"; exit 1 }
$envPath = Invoke-MsiQuery $msi "SELECT ``Value`` FROM ``Environment`` WHERE ``Environment``='envPath'"
if ($envPath.Count -ne 1 -or -not $envPath[0].EndsWith("[INSTALLFOLDER]")) { Write-Error "MSI check failed: PATH entry ($envPath)"; exit 1 }
$ctl = Invoke-MsiQuery $msi "SELECT ``FileName`` FROM ``File``"
if (-not ($ctl | Where-Object { $_ -like "*ymirctl.exe" })) { Write-Error "MSI check failed: ymirctl.exe is not installed"; exit 1 }
$sddl = Invoke-MsiQuery $msi "SELECT ``SDDLText`` FROM ``MsiLockPermissionsEx`` WHERE ``LockObject``='ProgramDataYmir'"
if ($sddl.Count -ne 1 -or -not $sddl[0].Contains("(A;OICI;FA;;;S-1-5-80-3634176441-2608231802-3936123067-875301740-2825028368)")) { Write-Error "MSI check failed: node directory ACL ($sddl)"; exit 1 }
Write-Host "MSI checks passed: service ymird as NT SERVICE\ymird with %ProgramData%\Ymir (ACL for that account), PATH entry, ymirctl.exe installed"
Write-Host "MSI created at $msi"
