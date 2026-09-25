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

param([int]$PublishAtMinute = 10, [string]$Project = "t2-sentiment-agent-live",
      [string]$Root = "")

$ErrorActionPreference = "Continue"
$root = if ($Root) { $Root } else { Split-Path -Parent $PSScriptRoot }
$public = Join-Path $root "public"
$stage = Join-Path $root ("var\site\" + $Project)
$logDir = Join-Path $root "var\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "site.log"

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

while ($true) {
    Wait-ForSlot
    try {
        if (Test-Path $stage) {
            Get-ChildItem $stage -Exclude ".vercel" | Remove-Item -Recurse -Force -Confirm:$false
        } else {
            New-Item -ItemType Directory -Force -Path $stage | Out-Null
        }
        Copy-Item -Path (Join-Path $public "*") -Destination $stage -Recurse -Force
        $leak = Get-ChildItem $stage -Recurse -File |
            Select-String -Pattern "Users\\\\prate|Users/prate|BITGET_SECRET|BITGET_PASSPHRASE" -List
        if ($leak) {
            Write-Log "refused: personal path or credential marker in $($leak[0].Path)"
        } else {
            Push-Location $stage
            $out = & vercel deploy --prod --yes --archive=tgz 2>&1 | Out-String
            Pop-Location
            $alias = ($out -split "`n" | Where-Object { $_ -match "Aliased" }) -join " "
            $generated = "unknown"
            try {
                $generated = (Get-Content (Join-Path $stage "summary.json") -Raw |
                    ConvertFrom-Json).generated_at
            } catch { }
            if ($alias) {
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
