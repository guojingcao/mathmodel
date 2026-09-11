# 代码改动一键提交脚本
# 用法:
#   .\commit.ps1 "本次改动说明"
#   .\commit.ps1 "本次改动说明" -Push
#
# 说明: 每次改写代码后运行, 便于出 bug 时回退。
#       未配置远程仓库时只做本地提交; 配置后加 -Push 会推送。
param(
    [Parameter(Mandatory = $true)][string]$Message,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1) 暂存(遵循 .gitignore, 大数据目录与 logs 不入库)
git add -A

# 2) 有实际改动才提交
$changed = git diff --cached --name-only
if (-not $changed) {
    Write-Host "[commit] 没有需要提交的改动。" -ForegroundColor Yellow
    exit 0
}
Write-Host "[commit] 本次改动文件:" -ForegroundColor Cyan
$changed | ForEach-Object { Write-Host "  $_" }

git commit -q -m $Message
$hash = (git rev-parse --short HEAD)
Write-Host "[commit] 已提交: $hash  $Message" -ForegroundColor Green

# 3) 可选推送
$remotes = git remote
if ($Push) {
    if ($remotes) {
        git push
        Write-Host "[commit] 已推送到远程。" -ForegroundColor Green
    } else {
        Write-Host "[commit] 未配置远程仓库, 无法推送。" -ForegroundColor Yellow
        Write-Host "         配置示例: git remote add origin <仓库URL>" -ForegroundColor Yellow
        Write-Host "                   git push -u origin main" -ForegroundColor Yellow
    }
} else {
    Write-Host "[commit] (未推送; 需要时加 -Push)" -ForegroundColor DarkGray
}

# 4) 显示最近提交, 便于回退
Write-Host "`n最近提交(回退用):" -ForegroundColor Cyan
git log --oneline -5
Write-Host "`n回退示例:  git revert <hash>   或   git reset --hard <hash>" -ForegroundColor DarkGray
