# Merge 27 group outputs + repair overrides -> all_papers_keyinfo.csv
$ErrorActionPreference = 'Stop'
$groupDir = Join-Path $PSScriptRoot '_tmp_groups'
$outCsv = 'D:\My_MathModeling_Project\corpus\03_analysis\all_papers_keyinfo.csv'

# 1. load all group outputs
$all = New-Object System.Collections.Generic.List[object]
foreach($i in 1..27){
  $f = Join-Path $groupDir ("out_g{0:D2}.json" -f $i)
  $arr = Get-Content $f -Encoding UTF8 -Raw | ConvertFrom-Json
  foreach($r in $arr){ $all.Add($r) }
}

# 2. apply repair overrides
$repairFile = Join-Path $groupDir 'repair_rows.json'
$repairMap = @{}
if(Test-Path $repairFile){
  foreach($r in (Get-Content $repairFile -Encoding UTF8 -Raw | ConvertFrom-Json)){
    $repairMap[$r.file_name] = $r
  }
}
foreach($row in $all){
  if($repairMap.ContainsKey($row.file_name)){
    $fix = $repairMap[$row.file_name]
    $row.core_model = $fix.core_model
    if(-not [string]::IsNullOrEmpty($fix.abstract_core)){ $row.abstract_core = $fix.abstract_core }
    if(-not [string]::IsNullOrEmpty($fix.keywords)){ $row.keywords = $fix.keywords }
  }
}

# 3. helpers
function Clean-Field([string]$v){
  if($null -eq $v){ return '' }
  $v = $v -replace "`r",'' -replace "`n",'' -replace '"',''
  return $v.Trim()
}
function Cap-Abstract([string]$v){
  if($v.Length -le 100){ return $v }
  $head = $v.Substring(0,100)
  $idx = $head.LastIndexOf([char]0x3002) # full-width 。
  if($idx -ge 0){ return $head.Substring(0,$idx+1) }
  return $head
}
function Csv-Esc([string]$v){
  if($v -match '[",]'){ return '"' + $v + '"' }
  return $v
}

# 4. clean each row
foreach($row in $all){
  $row.file_name = Clean-Field $row.file_name
  $row.year = Clean-Field $row.year
  $row.topic = Clean-Field $row.topic
  $row.abstract_core = Cap-Abstract (Clean-Field $row.abstract_core)
  $row.core_model = Clean-Field $row.core_model
  $kw = Clean-Field $row.keywords
  $toks = New-Object System.Collections.Generic.List[string]
  foreach($t in ($kw -split ',')){
    $tt = $t.Trim()
    if($tt -ne '' -and $tt.Length -gt 1){ $toks.Add($tt) }
  }
  $row.keywords = ($toks -join ',')
}

# 5. sort & validate
$sorted = @($all | Sort-Object -Property file_name)
$names = @($sorted | ForEach-Object { $_.file_name })
$dups = @($names | Group-Object | Where-Object { $_.Count -gt 1 })
$badLen = 0
$badYear = 0
$emptyCore = 0
$emptyAbs = 0
foreach($row in $sorted){
  if($row.abstract_core.Length -gt 100){ $badLen++ }
  $expYear = $row.file_name.Split('_')[0]
  if($row.year -ne $expYear){ $badYear++ }
  if([string]::IsNullOrWhiteSpace($row.core_model)){ $emptyCore++ }
  if([string]::IsNullOrWhiteSpace($row.abstract_core)){ $emptyAbs++ }
}

# 6. write CSV (UTF-8 BOM)
$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine('file_name,year,topic,abstract_core,core_model,keywords')
foreach($row in $sorted){
  $line = (Csv-Esc $row.file_name) + ',' + (Csv-Esc $row.year) + ',' + (Csv-Esc $row.topic) + ',' + (Csv-Esc $row.abstract_core) + ',' + (Csv-Esc $row.core_model) + ',' + (Csv-Esc $row.keywords)
  [void]$sb.AppendLine($line)
}
[System.IO.File]::WriteAllText($outCsv, $sb.ToString(), (New-Object System.Text.UTF8Encoding($true)))

Write-Output ("rows=" + $sorted.Count)
Write-Output ("dups=" + $dups.Count)
Write-Output ("abstract_over100=" + $badLen)
Write-Output ("year_mismatch=" + $badYear)
Write-Output ("empty_core_model=" + $emptyCore)
Write-Output ("empty_abstract_core=" + $emptyAbs)
Write-Output ("csv_saved_bytes=" + (Get-Item $outCsv).Length)
