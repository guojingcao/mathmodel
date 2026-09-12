# In-loop batch runner for Problem 3 (two-arm comparison).
# Usage: & 'problem3_robot\loop_batch.ps1' -Tag ring1150n9_a -Count 20 -RingR 1150 -RingN 9
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 parses .ps1 as ANSI/GBK).
param(
  [string]$Tag = "batch_a",
  [int]$Count = 12,
  [double]$RingR = 1150.0,
  [int]$RingN = 9,
  [string]$TeamNo = $env:ROBOT_ID,
  [int]$StartAt = 1,
  [switch]$NoLens,
  [switch]$NoProb
)
$env:PYTHONIOENCODING = "utf-8"
if (-not $TeamNo) { $TeamNo = "202604004013" }
$extra = @()
if ($NoLens) { $extra += "--no-lens" }
if ($NoProb) { $extra += "--no-prob" }
$ok = 0; $fail = 0
for ($i = $StartAt; $i -lt ($StartAt + $Count); $i++) {
  $t = "{0}_{1:d2}" -f $Tag, $i
  $t0 = Get-Date
  $out = python problem3_robot/robot.py --robot-id $TeamNo --tag $t --ring-r $RingR --ring-n $RingN @extra 2>&1
  $secs = [int]((Get-Date) - $t0).TotalSeconds
  $txt = ($out | Out-String)
  if ($txt -match "not started|connection refused|ERROR|Traceback" -or $txt -match [char]0x6D4B + [char]0x8BD5) {
    if ($txt -match [char]0x672A + [char]0x5F00 + [char]0x59CB) {
      Write-Output "[$t] ABORT: simulator session not active (click START in simulator) elapsed ${secs}s"
      $fail++
      break
    }
  }
  $vt = ($out | Select-String -Pattern "summary|final_virtual_time" | Select-Object -Last 1)
  $status = ($out | Select-String -Pattern "completed" | Select-Object -Last 1)
  Write-Output "[$t] done elapsed ${secs}s"
  $ok++
}
Write-Output "BATCH END: ok=$ok fail=$fail (tag=$Tag, r=$RingR, n=$RingN)"
