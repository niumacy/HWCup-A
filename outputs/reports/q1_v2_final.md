# Q1 v2 整改 — 最终交付报告

**生成时间**: 2026-09-25 17:50
**作者**: 成员 A
**状态**: ✅ 全部完成（commit 已落本地，push 受网络限制未执行）

---

## 一、问题清单（来自《Q1 整改方案》）

| # | 原始问题 | 修复状态 |
|---|---------|---------|
| 1 | 对齐是数学映射，没真实时间戳 | ✅ whisperx forced alignment |
| 2 | 没有 ASR 校验，无法验证 raw_text ↔ 音频一致性 | ✅ alignment JSON 含 asr_verification.wer |
| 3 | 没有 token-level 字段，B/C 无法做细粒度分析 | ✅ 4 个 per_token_* 字段 |
| 4 | 文档有 5 处夸大表述 | ✅ q1_decisions.md 5 处改写 |
| 5 | 没有 sha256 复现性 | ✅ extract_log.json 3 个 sha256 |
| 6 | 摘要表没有 effective length | ✅ q1_feature_summary.csv 28 列 |
| 7 | 没有对照案例（sync cases） | ✅ 3 个 case × 3 文件 + overview |
| 8 | paths.yaml 含完整硬编码路径 | ✅ ${REPO_ROOT} 占位符 |
| 9 | 没有 v1 → v2 版本说明 | ✅ q1_data_version.md |
| 10 | 没有验收清单 | ✅ outputs/reports/q1_v2_verification.json |

---

## 二、交付物清单

### 数据文件 (data/processed/features_q1/)
- ✅ `alignment/` — 100 个 JSON（含 ASR WER、词级时间戳、状态）
- ✅ `alignment_summary.json` — 100 条汇总 (aligned=62, partial=38, failed=0)
- ✅ `q1_features_all_v2.pkl` (17.76 MB) — v2 合并特征
- ✅ `q1_features_combined.pkl` → 软链接到 v2
- ✅ `backup_old/q1_features_all.pkl` — v1 备份
- ✅ `backup_old/extract_log_v1.json` — v1 日志
- ✅ `sync_cases/` — 3 个对照案例（normal / visual_missing / text_sparse）

### 脚本 (scripts/alignment/)
- ✅ `run_alignment.py` — 单条/批量对齐
- ✅ `build_v2.py` — 合并特征 + 软链接
- ✅ `build_sync_cases.py` — 3 个对照案例生成

### 工具 (src/utils/)
- ✅ `resolve_paths.py` — ${REPO_ROOT} 自动解析

### 文档 (docs/)
- ✅ `q1_data_version.md` — 新增（v1 ↔ v2 差异、字段表、回退方法）
- ✅ `feature_spec.md` — 升级 v2.0（新增 §10 v2 章节）
- ✅ `q1_decisions.md` — 5 处夸大修订 + 决策 #009

### 摘要表
- ✅ `outputs/tables/q1_feature_summary.csv` — 28 列，含 effective length

### 验收
- ✅ `outputs/reports/q1_v2_verification.json` — 10/10 通过
- ✅ `outputs/reports/q1_v2_final.md` — 本报告

### 配置
- ✅ `config/paths.yaml` — 脱敏为 ${REPO_ROOT}
- ✅ `.gitignore` — 更新

---

## 三、验收结果（10/10 通过）

```
✅ 1. 100 条 alignment JSON: 100 个
✅ 2. alignment_summary.json: aligned=62, partial=38, failed=0
✅ 3. q1_features_all_v2.pkl: 17.76 MB
✅ 4. q1_features_combined.pkl 软链接: → q1_features_all_v2.pkl
✅ 5. backup_old/ 含 v1: q1_features_all.pkl
✅ 6. v2 字段形状 + 新增字段: 新增 6 字段 (raw_text/annotation/4 个 per_token_*)
✅ 7. q1_feature_summary.csv 含 effective length: 列数=28
✅ 8. 3 个 sync cases 三件套: 3 个完整
✅ 9. extract_log.json 含 sha256: 3 个 sha256 字段
✅ 10. docs 修订 (v2.0 / 决策 #009 / 数据版本): 3 个文档均更新
```

---

## 四、关键统计

### 对齐质量
- 总样本: 100
- aligned (WER<0.3): 62
- partial (0.3<WER<0.6): 38
- failed: 0
- 平均 ASR WER: 0.394
- 平均匹配率: 0.606

### 数据规模
- v2 pkl: 17.76 MB (100 样本)
- alignment JSON: 1.6 MB (100 文件)
- 摘要 CSV: ~10 KB

### 时间戳精度
- audio: 0.01-0.02 秒（whisperx 输出）
- video: 1 帧（来自 fps × audio_sec）

---

## 五、Git 状态

```
commit 22554ee Q1 v2: 时序对齐升级 (whisperx forced alignment)
        126 files changed, 64367 insertions(+), 131 deletions(-)
        本地 commit 已落，origin push 待执行（需 GitHub 凭据）
```

### Push 阻塞说明
- 远程 origin: `https://github.com/niumacy/HWCup-A.git`
- 错误: `GnuTLS recv error (-110): The TLS connection was non-properly terminated`
- 原因: 无 GitHub 凭据 + 网络限制
- 解决: 在 `git push` 前先 `git remote set-url origin git@github.com:niumacy/HWCup-A.git` + 配置 SSH key，或手动在 GitHub 网页上传

---

## 六、与 B/C 的握手

### B (Q2 鲁棒预测)
- 加载方式不变: `from src.data_loader import load_q1_features`
- v1 / v2 文件名一致：`q1_features_combined.pkl`
- 100 条样本 ID 不变
- 4 个新字段可选忽略

### C (Q3 可解释预测)
- 新增 `per_token_*` 4 字段，可做 token-level 可解释性
- 新增 `src/utils/video_utils.py` 配合 alignment JSON 做关键帧抽取

---

## 七、风险与已知问题

1. **GitHub push 未执行**: 本地 commit 已落，需手动 push
2. **git 历史中含 niumacy 邮箱**: 在 commit ca08b27 / 82d4139 / 1feec8f 的 author 中
   - 影响: 仅 git log 可见，不会出现在 v2 数据中
   - 解决: 需要 `git filter-branch` 重写历史（破坏性操作，未执行）
3. **VAD stub**: pyannote/segmentation-3.0 是 gated 模型，使用 stub VAD
   - 影响: 对 forced alignment 结果无影响（wav2vec2 不依赖 VAD 分段）
4. **CPU 跑 whisperx**: GPU 不可用，CPU 模式约 12 分钟跑完 100 条
   - 影响: 单条对齐 ~7 秒，可接受
