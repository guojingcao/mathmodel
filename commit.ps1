# 代码改动一键提交并推送脚本
# 用法:
#   .\commit.ps1 "本次改动说明"              # 提交 + 推送(默认)
#   .\commit.ps1 "本次改动说明" -NoPush      # 只本地提交, 不推送
#
# 说明: 每次改写代码后运行, 便于出 bug 时回退。
param(
    [Parameter(Mandatory = $true)][string]$Message,
    [switch]$NoPush
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1) 暂存(遵循 .gitignore, 大数据目录与 logs 不入库)
git add -A

# 2) 有实际改动才提交
$changed = git diff --cached --name-only
if (-not $changed) {
    Write-Host "[commit] 没有需要提交的改动。" -ForegroundColor Yellow
} else {
    Write-Host "[commit] 本次改动文件:" -ForegroundColor Cyan
    $changed | ForEach-Object { Write-Host "  $_" }
    git commit -q -m $Message
    Write-Host ("[commit] 已提交: " + (git rev-parse --short HEAD) + "  " + $Message) -ForegroundColor Green
}

# 3) 推送到远程(默认执行)
if (-not $NoPush) {
    $remotes = git remote
    if ($remotes) {
        git push 2>&1 | ForEach-Object { Write-Host "  $_" }
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[push] 已推送到 origin/main" -ForegroundColor Green
        } else {
            Write-Host "[push] 推送失败, 请检查网络/SSH 密钥/仓库权限" -ForegroundColor Red
        }
    } else {
        Write-Host "[push] 未配置远程仓库, 跳过。" -ForegroundColor Yellow
    }
} else {
    Write-Host "[commit] (按 -NoPush 跳过推送)" -ForegroundColor DarkGray
}

# 4) 显示最近提交, 便于回退
Write-Host "`n最近提交(回退用):" -ForegroundColor Cyan
git log --oneline -5
Write-Host "`n回退示例:  git revert <hash>   或   git reset --hard <hash>" -ForegroundColor DarkGray
