$ErrorActionPreference = 'Stop'

$srcDir = 'D:\My_MathModeling_Project\workspace\scripts\_tmp_deepagg'
$files = @(
    'agg_batch01.json','agg_batch02.json','agg_batch03.json',
    'agg_batch04.json','agg_batch05.json'
)
$header = @('file_name','year','topic','batch','core_model','keywords',
            'abstract_paragraphs','abstract_structure','abstract_funcs',
            'sentence_classes','steps_missing','routine','notes','report_path')

$all = @()
foreach ($f in $files) {
    $path = Join-Path $srcDir $f
    $json = Get-Content -Raw -Encoding UTF8 $path | ConvertFrom-Json
    foreach ($obj in $json) {
        $row = [ordered]@{}
        foreach ($k in $header) {
            $v = $obj.$k
            if ($null -eq $v) { $v = '' }
            $s = [string]$v
            # strip leading/trailing whitespace
            $s = $s.Trim()
            # internal newlines -> space
            $s = $s -replace "`r`n", ' ' -replace "`n", ' ' -replace "`r", ' '
            # collapse multiple internal spaces is NOT required; keep as-is
            if ([string]::IsNullOrWhiteSpace($s)) { $s = '—' }
            $row[$k] = $s
        }
        $all += [pscustomobject]$row
    }
}

Write-Host "total objects: $($all.Count)"

# sort ascending by file_name
$sorted = @($all | Sort-Object -Property file_name)

function Format-CsvField([string]$val) {
    if ($val.Contains(',') -or $val.Contains('"')) {
        return '"' + $val.Replace('"','""') + '"'
    }
    return $val
}

$sb = New-Object System.Text.StringBuilder
[void]$sb.AppendLine(($header -join ','))
foreach ($row in $sorted) {
    $fields = foreach ($k in $header) { Format-CsvField ([string]$row.$k) }
    [void]$sb.AppendLine(($fields -join ','))
}

$outDir = 'D:\My_MathModeling_Project\corpus\03_analysis\deep_results'
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$outPath = Join-Path $outDir '深度精读分析总表.csv'
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText($outPath, $sb.ToString(), $enc)

Write-Host "written: $outPath"
Write-Host "file bytes: $((Get-Item $outPath).Length)"
