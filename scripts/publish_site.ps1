# Publishes the agent's public record (public/, written by `t2sa export` every hour) to its
# hosted page, so a judge always sees the live paper log without anything running on their side.
#
# Every hour, at PublishAtMinute past: copy public/ to a staging folder named after the Vercel
# project, refuse to publish if it contains a personal path or a credential marker, then deploy it.
# Every attempt is logged to var/logs/site.log.
#
# Why a fixed minute and not "every 3600 s": `t2sa export` rewrites public/ at the top of each hour
# and takes tens of seconds. A publisher started near the hour copied the folder before the export
# finished, so every publish served the previous hour's record (seen 2026-09-24: published 19:00:59,
# summary generated 18:00:38). Publishing at :10 leaves the export ten minutes, and the log records
# the summary's own generated_at so a stale publish is visible rather than silent.

# -Project names the hosted page (the Vercel project the staging folder is linked to). Each paper
# run publishes to its own, so run 2 never overwrites run 1's record:
#   run 1: t2-sentiment-agent-live (the default)    run 2: -Project t2-sentiment-agent-run2

# -Root names the agent root whose public/ is published; it defaults to this script's own checkout.
# Run 1 is published by this script from run 2's checkout with -Root pointing at run 1's root, so
# run 1's working tree is never edited while its window is open.
#
# Why --archive=tgz, and why the deploy result is checked (2026-09-25): run 1's record passed
# 15,000 files (the evidence blobs) and Vercel began rejecting every upload with "files should NOT
# have more than 15000 items"; the loop still logged "published", so the hosted page silently
# froze at 09:00 UTC for ten hours. One archive is one file, and a deploy without an "Aliased" line
# is now logged as a failure.
#
# Why the copy holds the export lock, copies only what changed, and is checked before it deploys
# (2026-09-26): run 1's record reached 27,000 blobs, its export began re-moving every one of them
# each hour, and the export ran past :10. Copy-Item then held files open while the export was
# replacing them, and Windows refused the replace ("[WinError 5] Access is denied"). The export
# died half-way, twice: a new ledger.jsonl went up, the blobs it cites and the summary did not, and
# this script deployed that mixture for three hours while logging the 22:00 record as published.
# Now:
#   * the copy takes the same OS lock `t2sa export` holds for the whole export
#     (var/run/export-<Mode>.lock, the byte at offset 65536; runtime/wiring.py InstanceLock), so the
#     two never touch public/ at once. It waits while an export runs, up to -LockWaitMinutes;
#   * robocopy /MIR copies only files that changed, so the lock is held for seconds, not minutes;
#   * the copy is deployed only when summary.json's ledger head equals ledger.jsonl.head's, which
#     holds only once an export has finished moving every file (site/export.py publishes the
#     summary last), and only when that record is newer than the one last deployed.

param([int]$PublishAtMinute = 10, [string]$Project = "t2-sentiment-agent-live",
      [string]$Root = "", [string]$Mode = "paper", [int]$LockWaitMinutes = 45)

$ErrorActionPreference = "Continue"
$root = if ($Root) { $Root } else { Split-Path -Parent $PSScriptRoot }
$public = Join-Path $root "public"
$stage = Join-Path $root ("var\site\" + $Project)
$logDir = Join-Path $root "var\logs"
$lockPath = Join-Path $root ("var\run\export-" + $Mode + ".lock")
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "site.log"
$lastPublished = $null

function Write-Log([string]$text) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $log -Value "$stamp $text"
}

function Wait-ForSlot {
    $now = (Get-Date).ToUniversalTime()
    $slot = $now.Date.AddHours($now.Hour).AddMinutes($PublishAtMinute)
    if ($slot -le $now) { $slot = $slot.AddHours(1) }
    Start-Sleep -Seconds ([int][Math]::Ceiling(($slot - $now).TotalSeconds))
}

function Enter-ExportLock {
    # Returns the open, locked stream, or $null when an export held the lock for the whole wait.
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $lockPath) | Out-Null
    $deadline = (Get-Date).AddMinutes($LockWaitMinutes)
    while ($true) {
        $stream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::ReadWrite)
        try {
            $stream.Lock(65536, 1)
            $holder = @{ pid = $PID; host = [System.Net.Dns]::GetHostName(); mode = $Mode;
                started_at = (Get-Date).ToUniversalTime().ToString("o");
                by = "scripts/publish_site.ps1 copying public/" } | ConvertTo-Json -Compress
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($holder + "`n")
            $stream.SetLength(0)
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Flush()
            return $stream
        } catch [System.IO.IOException] {
            $stream.Dispose()
            if ((Get-Date) -ge $deadline) { return $null }
            Start-Sleep -Seconds 5
        }
    }
}

function Exit-ExportLock($stream) {
    try { $stream.Unlock(65536, 1) } catch { }
    $stream.Dispose()
}

function Read-Json([string]$path) {
    try { return (Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

function Invoke-PublishOnce {
    # One publish: copy under the export lock, check the copy is one finished export and a new
    # one, scan it, deploy it. Every outcome is one line in var/logs/site.log.
    try {
        # The staging copy doubles public/ on disk, and on 25 Sep 2026 the disk reached 0 bytes
        # free. Under 5 GB every publish says so; under 1 GB it refuses to copy rather than fill
        # the disk the ledger appends to.
        $freeBytes = (Get-PSDrive -Name (Split-Path -Qualifier $public).TrimEnd(':')).Free
        if ($freeBytes -lt 1GB) {
            Write-Log ("not published: DISK LOW, {0:N1} GB free, under the 1 GB floor" -f ($freeBytes / 1GB))
            return
        }
        if ($freeBytes -lt 5GB) {
            Write-Log ("DISK LOW: {0:N1} GB free, under 5 GB" -f ($freeBytes / 1GB))
        }
        New-Item -ItemType Directory -Force -Path $stage | Out-Null
        $lock = Enter-ExportLock
        if ($null -eq $lock) {
            Write-Log "not published: an export held $lockPath for $LockWaitMinutes minutes"
            return
        }
        try {
            & robocopy $public $stage /MIR /XD .vercel /R:5 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Null
            $copied = $LASTEXITCODE
        } finally {
            Exit-ExportLock $lock
        }
        if ($copied -ge 8) {
            Write-Log "not published: robocopy failed copying public/ (exit $copied)"
            return
        }
        $summary = Read-Json (Join-Path $stage "summary.json")
        $anchor = Read-Json (Join-Path $stage "ledger.jsonl.head")
        $generated = if ($summary) { $summary.generated_at } else { "unknown" }
        $summaryHead = if ($summary) { $summary.ledger.head_hash } else { $null }
        $anchorHead = if ($anchor) { $anchor.head_hash } else { $null }
        if (-not $summaryHead -or $summaryHead -ne $anchorHead) {
            Write-Log ("not published: the copy is not one finished export (summary head " +
                "$summaryHead, ledger head $anchorHead, record generated $generated)")
            return
        }
        if ($generated -eq $script:lastPublished) {
            Write-Log "not redeployed: no new record since the one generated $generated"
            return
        }
        $leak = Get-ChildItem $stage -Recurse -File |
            Where-Object { $_.FullName -notlike "*\.vercel\*" } |
            Select-String -Pattern "Users\\\\prate|Users/prate|BITGET_SECRET|BITGET_PASSPHRASE" -List
        if ($leak) {
            Write-Log "refused: personal path or credential marker in $($leak[0].Path)"
        } else {
            Push-Location $stage
            $out = & vercel deploy --prod --yes --archive=tgz 2>&1 | Out-String
            Pop-Location
            $alias = ($out -split "`n" | Where-Object { $_ -match "Aliased" }) -join " "
            if ($alias) {
                $script:lastPublished = $generated
                Write-Log ("published record generated $generated " + $alias.Trim())
            } else {
                $why = ($out -split "`n" | Where-Object { $_ -match "error|Error|message" } |
                    Select-Object -First 2) -join " "
                Write-Log ("publish FAILED for record generated $generated -- " + $why.Trim())
            }
        }
    } catch {
        Write-Log "publish failed: $($_.Exception.Message)"
    }
}

function Start-Publishing {
    while ($true) {
        Wait-ForSlot
        Invoke-PublishOnce
    }
}

# Dot-sourced (tests/runtime/test_export_lock.py), the script only defines its functions.
if ($MyInvocation.InvocationName -ne ".") { Start-Publishing }
