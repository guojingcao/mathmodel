# =====================================================================
# configure_dsh_origin_mcp.ps1
# One-shot, idempotent setup of the dsh-origin-plugin MCP server entry
# inside the DSH "web" profile. Safe to run repeatedly; detects the
# actual install each time and repairs whatever is missing.
#
# Steps:
#   [1/4] Detect the dsh-origin-plugin package (searches the candidate
#         locations and keeps the first one that contains
#         origin_mcp_server.py).
#   [2/4] Ensure a Python 3.10+ virtualenv (default:
#         %USERPROFILE%\dsh_origin_plugin\.venv). If missing, create it
#         and install mcp / originpro / pywin32 / numpy.
#   [3/4] Write or refresh a config-only "mcp-origin" override in
#         %USERPROFILE%\.dsh\profiles\web\cordis.patch.yml
#         (2-space indentation, forward-slash paths, guarded by marker
#         comments so it never duplicates the loader entry id).
#   [4/4] Validate by dumping the merged config and grepping mcp-origin.
#
# Run it with:
#   powershell -ExecutionPolicy Bypass -File configure_dsh_origin_mcp.ps1
# =====================================================================

$ErrorActionPreference = 'Stop'

Write-Host '== [1/4] Detecting dsh-origin-plugin installation =='
$user = $env:USERPROFILE
$candidates = @(
    (Join-Path $user '.dsh\profiles\web\node_modules\dsh-origin-plugin'),
    (Join-Path $user 'dsh_origin_plugin')
)
$pluginDir = $null
foreach ($c in $candidates) {
    if (Test-Path -LiteralPath (Join-Path $c 'origin_mcp_server.py')) {
        $pluginDir = $c
        break
    }
}
if (-not $pluginDir) {
    throw 'dsh-origin-plugin not found. Install the bundle first, e.g.:  dsh plugin --profile web add dsh-origin-plugin'
}
$serverScript = Join-Path $pluginDir 'origin_mcp_server.py'
Write-Host ("  plugin dir    : {0}" -f $pluginDir)
Write-Host ("  server script : {0}" -f $serverScript)

# --------------------------------------------------------------- [2/4]
Write-Host '== [2/4] Ensuring Python virtualenv =='
$venvDir = Join-Path $user 'dsh_origin_plugin\.venv'
$venvPy  = Join-Path $venvDir 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    $py = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $py) {
        $pyLauncher = (Get-Command py -ErrorAction SilentlyContinue).Source
        if (-not $pyLauncher) { throw 'No Python 3.10+ found on PATH. Install Python first, then re-run.' }
        $py = "$pyLauncher -3"
    }
    Write-Host ("  creating venv: {0}" -f $venvDir)
    & $py -m venv $venvDir
    if (-not (Test-Path -LiteralPath $venvPy)) { throw 'venv creation failed' }
} else {
    Write-Host ("  venv present  : {0}" -f $venvPy)
}
Write-Host '  installing python deps: mcp originpro pywin32 numpy'
& $venvPy -m pip install --disable-pip-version-check --upgrade pip
& $venvPy -m pip install --disable-pip-version-check mcp originpro pywin32 numpy
if ($LASTEXITCODE -ne 0) { throw 'pip install failed' }
& $venvPy -c "import mcp, originpro, win32com.client, numpy; print('  deps import OK')"

# --------------------------------------------------------------- [3/4]
Write-Host '== [3/4] Writing mcp-origin override into profile cordis.patch.yml =='
$profile   = Join-Path $user '.dsh\profiles\web'
$patchFile = Join-Path $profile 'cordis.patch.yml'
if (-not (Test-Path -LiteralPath $patchFile)) {
    throw "profile patch missing: $patchFile"
}
$open  = '# --- dsh-origin-plugin'
$close = '# --- end dsh-origin-plugin'
$p = ($venvPy -replace '\\', '/')
$s = ($serverScript -replace '\\', '/')
$block = @"
$open (Origin plotting MCP server; config-only override of the bundle mcp-origin entry)
- id: mcp-origin
  config:
    serverName: origin
    transport: stdio
    command: '$p'
    args: ['-u', '-X', 'utf8', '$s']
    env:
      PYTHONIOENCODING: utf-8
      PYTHONUNBUFFERED: '1'
    failOnStartupError: false
    toolCallTimeoutMs: 120000
$close
"@

$content = [System.IO.File]::ReadAllText($patchFile, [System.Text.Encoding]::UTF8)
# 1) remove an empty "[]" array body, if present
$content = $content -replace '(?ms)^[ \t]*\[\][ \t]*(\r?\n)?', ''
# 2) remove any previous marked block of this script (idempotent refresh)
$escOpen  = [regex]::Escape($open)
$escClose = [regex]::Escape($close)
$content  = [regex]::Replace($content, "(?ms)[ \t]*$escOpen.*?$escClose[ \t]*\r?\n?", '')
# 3) guard: an UNMARKED mcp-origin entry elsewhere would duplicate the loader id
if ($content -match '(?m)^[ \t]*-[ \t]+id:[ \t]+mcp-origin[ \t]*$') {
    throw 'cordis.patch.yml already holds an unmarked mcp-origin entry; remove it first (a second full entry breaks DSH startup with "duplicate loader entry id"), then re-run.'
}
$content = $content.TrimEnd() + "`r`n`r`n" + $block + "`r`n"
[System.IO.File]::WriteAllText($patchFile, $content, (New-Object System.Text.UTF8Encoding($false)))
Write-Host ("  patch updated : {0}" -f $patchFile)

# --------------------------------------------------------------- [4/4]
Write-Host '== [4/4] Validating merged config =='
& dsh --profile web --dump-config 2>$null | Select-String -Pattern 'mcp-origin|origin_mcp_server|serverName' | Select-Object -First 12

Write-Host ''
Write-Host 'DONE. Next steps:'
Write-Host '  1) Restart dsh web.'
Write-Host '  2) In a new chat ask: "Use Origin to plot y = x^2 and export a PNG."'
Write-Host '  3) If the mcp__origin__* tools are missing, check:'
Write-Host '       dsh --profile web --dump-config | Select-String mcp-origin'
