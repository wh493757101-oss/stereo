# material_existing_data_v1 — 用户手动执行命令（PowerShell / VS Code）
#
# 本文件由 Agent 生成/更新（2026-09-30 修复轮）。Agent 未执行、也不会执行
# 其中任何 --fit 命令；训练由用户手动运行。
# 启动方式：& .\manual_commands.ps1（不要用 python 启动 .ps1）。
# 默认行为：不带 -Mode 参数时只做 dry-run：只读输入，但写入新的准入报告。
# 明确传 -Mode fit / -Mode fold0 / -Mode p2 才会真正训练。
#
# 产物路径：
#   协议（沿用上轮，已验证）：
#     analysis\runs\material_existing_data_v1\20260930T021500Z_a3f7c1\protocol\protocol.json
#   修复轮升级后的缓存（本轮新目录，绑定+统一准入已通过，字节与上轮一致）：
#     analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\upgrade\corrected_features
#   修复轮图册（从已保存证据重绘，68 面板）：
#     analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\atlas_rebuild\atlas\

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('dryrun', 'fit', 'fold0', 'p2')]
    [string]$Mode = 'dryrun'
)

conda activate stereo
Set-Location 'D:\Python\Project\my_data'

# ---- 解释器核对（必须输出 D:\Python\CondaPkgs\stereo\python.exe）----
$py = python -c "import sys; print(sys.executable)"
if ($py -ne 'D:\Python\CondaPkgs\stereo\python.exe') {
    Write-Error "interpreter mismatch: $py — aborting (fix conda env first)"
    exit 1
}
Write-Host "interpreter OK: $py"

$PROTOCOL = 'analysis\runs\material_existing_data_v1\20260930T021500Z_a3f7c1\protocol\protocol.json'
$FEATURES = 'analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\upgrade\corrected_features'

# ---- preflight：准入 + 预览（只读输入、写新报告；失败停止，不得继续 fit）----
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$preflight = "analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\preflight_$stamp"
python scripts/train_material_probe.py `
    --protocol $PROTOCOL `
    --features $FEATURES `
    --output-dir $preflight `
    --dry-run
if ($LASTEXITCODE -ne 0) {
    Write-Error "preflight FAILED (exit $LASTEXITCODE) — fitting is not allowed; inspect stderr and bring the output back for review"
    exit $LASTEXITCODE
}
Write-Host "preflight passed: $preflight\report.json"
Write-Host "NOTE: a passing dry-run is admission only — it is NOT itself the training run."

switch ($Mode) {
    'dryrun' {
        Write-Host "Mode=dryrun: nothing further runs. For real training re-run with -Mode fit (all folds), -Mode fold0 or -Mode p2."
    }
    'fit' {
        # P0/P1/P2 x 全部 3 折（约数分钟，CPU）
        python scripts/train_material_probe.py `
            --protocol $PROTOCOL `
            --features $FEATURES `
            --fold all `
            --variants P0 P1 P2 `
            --output-dir "analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\probe_fit_$stamp" `
            --fit
        if ($LASTEXITCODE -ne 0) { Write-Error "fit failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
        Write-Host "summary: analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\probe_fit_$stamp\summary.json"
    }
    'fold0' {
        # 只跑 fold0 的三个变体（安全续跑示例：换新输出目录，永不覆盖）
        python scripts/train_material_probe.py `
            --protocol $PROTOCOL `
            --features $FEATURES `
            --fold fold0 `
            --variants P0 P1 P2 `
            --output-dir "analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\probe_fit_fold0_$stamp" `
            --fit
        if ($LASTEXITCODE -ne 0) { Write-Error "fit failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    }
    'p2' {
        # 只跑 P2 的全部折（安全续跑示例：换新输出目录，永不覆盖）
        python scripts/train_material_probe.py `
            --protocol $PROTOCOL `
            --features $FEATURES `
            --fold all `
            --variants P2 `
            --output-dir "analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\probe_fit_p2_$stamp" `
            --fit
        if ($LASTEXITCODE -ne 0) { Write-Error "fit failed (exit $LASTEXITCODE)"; exit $LASTEXITCODE }
    }
}

# ---- 结果查看（fit 完成后手动执行）----
# Get-Content "analysis\runs\material_existing_data_v1\20260930T045500Z_c5d9e8\validation\probe_fit_<stamp>\summary.json"
# 关注：oof.P0/P1/P2.macro_f1_mean；paired.P2_minus_P1 / P2_minus_P0（fixed/broke）；
#       per_group；key_class_pairs（plastic_fish vs real_fish、metal vs plastic submarine）。
#
# 中断续跑：入口每次从头训练所请求的 fold/variant（全批次单折通常 < 1 分钟），
# 用新输出目录重跑即可；已写出的目录永不覆盖。失败/中断目录可能不完整，
# 应保留用于诊断，不把它作为成功训练或可续训检查点。
#
# 诚实边界：本协议只保证训练层组隔离，不构成新的独立泛化证据；
# P2 相对 P1/P0 的差值只在本协议开发池内成立，不据此声称偏振收益。
