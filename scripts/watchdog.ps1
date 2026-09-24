# Keeps the paper run alive on an always-on machine.
#
# `t2sa go-live` proves the Demo key, writes the genesis once, and runs the loop; run again after a
# stop, it resumes the same ledger. This script restarts it after any exit except a refusal
# (exit 3: the environment or key is not Demo) or a usage error (exit 2), which need a person.
# Every start and stop is appended to var/logs/watchdog.log.

param([int]$PauseSeconds = 60)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$t2sa = Join-Path $root ".venv\Scripts\t2sa.exe"
$logDir = Join-Path $root "var\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "watchdog.log"
$env:PYTHONUTF8 = "1"

function Write-Log([string]$text) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $log -Value "$stamp $text"
}

while ($true) {
    Write-Log "starting t2sa go-live"
    $out = Join-Path $logDir ("run-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ") + ".log")
    & $t2sa --root $root go-live *>> $out
    $code = $LASTEXITCODE
    Write-Log "t2sa go-live exited with code $code (output: $out)"
    if ($code -eq 2 -or $code -eq 3) {
        Write-Log "not restarting: exit $code needs a person (usage error or refused environment)"
        break
    }
    Start-Sleep -Seconds $PauseSeconds
}
