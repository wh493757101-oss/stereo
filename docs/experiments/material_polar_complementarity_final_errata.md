# material_polar_complementarity_v1 — 最终文字勘误（2026-10-12）

本勘误只订正文字，不改代码、不重算统计、不运行实验。统计数字复用已
验收的 `review_closeout/20261012T025124Z_83286b31` 产物；工程与诊断收
尾已完成，提交 Codex 最终文字复核。科学结论保持：**尚未建立偏振净收
益；路线 C 为当前审查结论，现有证据不足以确认空间互补性或测量侧主导
解释**。本勘误在以下四项上取代此前的不准确表述。

## 勘误一：守恒状态展示取值错误

旧文字导出报告写了“layered_subsets.json 守恒校验 None”。正确来源为
[layered_subsets.json](../../analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T025124Z_83286b31/postprocess/layered_subsets.json)
→ `layered_subsets.conservation_checks.verdict`，实际值为 **PASS**。
这是展示取值错误（报告生成时从错误层级取值），统计本身已验收；本次
不重算统计，也不表述为守恒校验曾失败。

## 勘误二：pc003 日志字节数

此前完成报告中的字节数有误，归档清单本身正确。以
[archive_manifest.json](../../analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T032300Z_aa394db6/retained_test_diagnostics/pc003/archive_manifest.json)
为准：

| 文件 | 字节数 |
|---|---|
| run1_log.txt | 7323 |
| run2_log.txt | 5253 |
| run3_log.txt | 3236 |
| run4_log.txt | 2198 |
| run7_log.txt | 102 |

## 勘误三：缺失日志的表述

run5_log.txt、run6_log.txt 的准确表述是：**本次归档时未找到，未纳入
归档；不能据此判断是否曾产生或是否执行。** 此前完成报告与归档清单中
“从未产生”“not produced by the earlier runs”的断言属于过度推断，现予
撤回。归档清单为历史文件，保留原文不改；本节明确订正其解释。

## 勘误四：环境恢复证据的强度

[environment_restore.json](../../analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T032300Z_aa394db6/environment_restore.json)
保存了实际环境值并记录 `restored_values_match_saved=true`，可证明首次
运行的保存、try/finally 恢复及逐值匹配。
[environment_restore_followup.json](../../analysis/runs/material_polar_complementarity_diagnostic_v1/review_closeout/20261012T032300Z_aa394db6/environment_restore_followup.json)
只有说明性声明，没有同等的实际前后值及匹配核验记录，不能证明同等细
节。准确表述：**首次记录包含实际值及恢复匹配结果；后续记录为说明性
声明，无法从该文件独立核验同等细节。** 不反向推断后续环境没有恢复，
也不补造历史记录。
