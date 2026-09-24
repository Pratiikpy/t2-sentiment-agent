# Publishes the agent's public record (public/, written by `t2sa export` every hour) to its
# hosted page, so a judge always sees the live paper log without anything running on their side.
#
# Every PublishEvery seconds: copy public/ to a staging folder named after the Vercel project,
# refuse to publish if it contains a personal path or a credential marker, then deploy it.
# Every attempt is logged to var/logs/site.log.

param([int]$PublishEvery = 3600)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$public = Join-Path $root "public"
$stage = Join-Path $root "var\site\t2-sentiment-agent-live"
$logDir = Join-Path $root "var\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "site.log"

function Write-Log([string]$text) {
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    Add-Content -Path $log -Value "$stamp $text"
}

while ($true) {
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
            $out = & vercel deploy --prod --yes 2>&1 | Out-String
            Pop-Location
            $alias = ($out -split "`n" | Where-Object { $_ -match "Aliased" }) -join " "
            Write-Log ("published " + $alias.Trim())
        }
    } catch {
        Write-Log "publish failed: $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $PublishEvery
}
