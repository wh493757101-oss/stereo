# material_conditional_residual_v1 — P0 输出上下文的条件修正（预注册）

状态：**设计冻结；入口与合成验收完成；真实 dry-run 通过；真实 fit 由用户
执行（18 次小型拟合）**。

## 1. 问题与设计

问题：偏振修正是否需要灰度上下文？本批只实现**一种**小容量条件修正：
上下文仅来自冻结 P0 自身的输出——4 个 softmax 值 + 最大/次大 softmax 差
（模型输出特征，**不称校准概率**，不做标准化），与冻结局部版本 C14/S16
拼接后经 `Linear(35, 8) → SiLU → Linear(8, 4, bias=False)`（末层零初始
化、前层按每折拟合种子初始化）。

```
delta = valid_s * 0.5 * tanh(head(input))    # |delta| ≤ 0.5
final_logits = P0_logits + delta
loss = CE(final_logits, label) + 0.01 * mean(delta²)
```

其余全部冻结：三折、P0 及其 scaler（含运行时身份校验）、局部特征定义
`local_partition_interior_band_v1`（C14/S16）、200 epoch、CPU/full-batch、
AdamW lr=0.01、wd=0.001、epoch0 候选、inner_val 选模（外层只评一次）、
缺失 S 精确回退、RC（S 槽置零）/RCS 对照、打乱 v2、六条门槛。无 GT 路
由、不锁类别组、不解冻 P0、不做结构/超参搜索。

种子 2026/2027/2028：2 变体 × 3 折 × 3 种子 = **18 次小型拟合**，全部
报告；不挑最佳种子；多种子是重复拟合而非独立采集样本，不宣称独立泛化。

## 2. 报告口径

- 固定基线（R0）、线性修正（linear_reference，指向身份修复后的线性局
  部拟合 summary；缺失时记 NOT_AVAILABLE）、本批非线性修正；
- RC−R0 与 RCS−RC 的增量分列；per_class/per_group 按比较分列、跨种子
  合计并与其 paired 总数核对；
- 全部种子的均值与范围、各折/各类/各组的改对改错；
- 内容打乱诊断 v2（复合分箱、完整 S16 同 donor）在每个训练种子的冻结
  RCS 头上运行（3 种子 × 3 折 × 5 内容种子 = 45 份）。

## 3. 预先冻结的推进门槛与停止条件

沿用六条工程推进门槛（对三种子均值口径）：RCS 三折平均 Macro-F1 同时
高于 R0 与 RC；至少两折不下降；最差折下降 ≤1pp；跨种子合计外层净改对；
各类召回下降 ≤1pp；至少两个采集组净改对。

**停止条件（预注册）**：若条件修正相对 RC 与 R0 仍无稳定净收益，明确
建议**停止本轮偏振模型扩张**；不得自行进入更大网络或 joint 微调。

## 4. 实现与验收

- `core/material_conditional_residual.py`：上下文（确定性 softmax）、
  条件头（末层零初始化）、装配（RC 置零 S 槽、缺失行 S 零但上下文保
  留、NaN 拒绝）、训练循环（无 outer_eval 参数、epoch0 候选、共享选模
  规则）。
- `scripts/train_material_conditional_residual.py`：复用冻结准入链 + 局
  部缓存核验 + P0 身份校验（oracle jsonl 字节 SHA、逐样本 logits
  atol 1e-4、三处禁用槽精确为零）；R0 logits/predictions/指标全部来自
  同一已核验运行时链；诊断失败落盘 FAILED+failed_phase。
- 验收：32 项合成测试（含 18 次拟合 + 45 份打乱的完整 main --fit、拒
  绝覆盖、故障注入）；真实 dry-run exit 0。

## 5. 四项验收缺口修复（2026-10-09，cond_fix 批）

1. **实际 RC 路径未屏蔽 S**：run_fit 每折只以 variant="RCS" 构建一次
   fold_inputs，RC 直接复用同一张量。修复：`_fold_conditional_inputs`
   只产出变体共享部分（标准化 C/S、已核验 P0 logits、上下文、标签、行
   序），新增 `_variant_inputs(shared, variant)` 按 RC/RCS 分别装配；
   RC 的 S 槽 [14:30] 在 train/inner_val/outer_eval 精确为零。训练器捕
   获探针（包装原训练函数、先记录再委托）证明 18 次调用的输入满足变体
   语义；扰动合成 S 后 RC 前向输出不变。
2. **paired 嵌套列表**：旧 paired 循环 extend 的是整折列表，zip 截断
   （六全对样本被计成 broke=3/both_correct=3）。修复：新增
   `align_paired_samples`（按 sample_name 对齐逐样本记录，拒绝重复/缺
   失/额外样本与 class/group 矛盾，LocalMismatch）+ 支持数不变量
   （fixed+broke+both_correct+both_wrong == 该种子外层支持数；支持数由
   冻结协议派生，不硬编码 2446）；paired.total 注明为"样本-模型对"计
   数（跨种子是重复拟合，不是 7338 个独立样本）。
3. **线性参照绑定**：配置现绑定已完成的
   `manual_20261009T135646/summary.json`（SHA256 22ff7a81…，与预期一
   致），准入（含 dry-run）核验 mode=fit、COMPLETED、实验与 region 规
   则、协议/缓存定义/折/类别顺序、P0 oracle 身份逐折一致后才放行；文
   件缺失、SHA 或身份冲突在条件头训练前拒绝（错误历史拟合
   manual_20261009T130447 因 SHA 与 P0 oracle 身份不符被自动拒绝）。
4. **折均值 vs pooled**：`paired.*.pooled_macro_f1` 实为折均值的种子平
   均——已删除该误标字段，改为 `paired.*.macro_f1_by_variant`（每变体
   分列 `fold_mean_of_seeds` 与 `pooled_mean_of_seeds`），
   `variants.*` 新增 `pooled_macro_f1_mean_of_seeds`；手动脚本结果阅读
   说明同步。合成例子（各折支持数不等）验证两口径确实不同，并从逐样本
   记录用独立 F1 复算（布尔计数，不经生产汇总函数）核对两者。

另：P0 目录不变测试改为真实前后对比（拟合前在夹具内捕获文件集合/
SHA256/长度，拟合后独立复算并要求完全相等），不再是仅拟合后计数。
已有回归（x_eval、metrics 契约、比较分列、数组准入、打乱 v2、失败状
态）全部保留。真实 18 次拟合 NOT_RUN。

## 6. 最后两项验收缺口修复（2026-10-09，cond_fix2 批）

1. **expected_sha256 必填且严格校验**：旧实现
   `if expected_sha and sha != expected_sha` 使缺键、`""`、`None`、
   `False` 完全跳过摘要比较，真值非字符串（True/数字/列表/字典）则落
   入 MISMATCH 语义。修复：expected_sha256 必须存在且为恰好 64 个十六
   进制字符的字符串（大小写均可，格式验证后统一小写比较）；格式非法按
   拒绝语义（exit 1，stderr 含 expected_sha256），格式合法但内容不符按
   MISMATCH 语义（exit 2）。不做 str()/bool() 强转，不以当前文件摘要
   补填预期值。校验位于准入内、输出目录创建与任何残差头构建/拟合之前
   （拒绝时 output_dir 根本不产生，训练调用数为 0——由先记录再抛错的
   训练探针证明）。原有 mode/status/experiment/region/协议/缓存定义/
   折/P0 oracle 身份检查全部保留；既有"状态未完成/P0 冲突"测试的合成
   配置已补上合法摘要，确保仍命中各自目标检查而非被新格式门遮蔽。
   19 个参数化反例（缺键、""、None、False、True、0、12345、0.5、[]、
   列表、{}、字典、63/65 长度、非十六进制、0x 前缀、前后空白、内嵌换
   行）+ 大写合法绑定 + 合法匹配 + 合法但不匹配（exit 2）全覆盖。
2. **打乱诊断绑定已保存 RCS 预测**：重复前向自洽不等于与正式评估路径
   一致。新增 `verify_normal_matches_saved_rcs`：每折 × 训练种子的诊断
   在 normal_logits 计算与自洽检查之后、五次内容打乱之前，读取同一
   拟合保存的 `models/seed_<seed>/<fold>_RCS/outer_eval_predictions.jsonl`
   （先经 fold_variant_report.json 核对 fold/variant=RCS/seed 身份，拒
   绝错用 RC、其他折或其他种子），按 sample_name 对齐（拒绝重复/缺失/
   额外，期望集合来自冻结样本身份而非待核对文件），核对 class_id/group
   与冻结身份一致、两边 prediction 各自 == argmax(logits)、logits 有限
   四维、normal_logits 与保存 logits 逐样本精确相等（np.array_equal，
   float32 同路径，不用 np.allclose；出现误差时报告最大绝对误差而非调
   容差）、normal_prediction 与保存 prediction 一致。守卫只读文件，不
   增加模型前向次数；失败沿用 FAILED + failed_phase=shuffle_diagnostic、
   非零退出、不发布 COMPLETED、保留拟合产物。故障注入经真实合成
   main --fit 验证：篡改一份已保存 RCS logits（整体 +0.25，argmax 不
   变，仅精确绑定能捕获）→ 运行非零、summary FAILED、
   failed_phase=shuffle_diagnostic、无 shuffle 字段、拟合产物保留。
   测试侧以独立名称映射/数组比较/argmax 复算全部 9 个 RCS 模型 ×
   45 份打乱报告的绑定，不调用生产守卫证明生产守卫。

验证（cond_fix2 批）：RED 39 项预期失败 → GREEN 47 项通过；两测试文件
90 项通过；聚焦 7 文件 256 项通过；全仓 2231 通过 / 1 失败（既有
LICENSE CRLF/LF，如实单列）/ 6 跳过；compileall 0；git diff --check 0；
真实 dry-run exit 0（严格 SHA 校验绑定 22ff7a81…，P0 oracle match，
三折 P0 身份零误差）。真实 18 次拟合仍 NOT_RUN（用户手动执行）。
