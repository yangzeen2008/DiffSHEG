# 跨模态时间轴修复与训练准入记录

- 记录 ID：R-2026-08-30-004
- 日期：2026-08-30
- 状态：数据与缓存修复已验证；5-epoch 全量试跑已通过；正式 500-epoch 重训已排队，等待既有全验证自然结束后自动启动
- 新数据版本：`beat_4english_15_141_sync_v1`
- 试跑实验：`beat_FM_aa_x0_aligned_sync_pilot_v1`

## 结论

此前“表情与语音完全对不上”不是固定延迟或 Blender 渲染问题，而是训练数据的时间尺度错误：原始 BVH 为约 120 FPS、面部 JSON 为 60 FPS，但旧预处理直接复制全部帧，数据集随后把两者都按 15 FPS 索引。约 69 秒样例因此包含 8280 个动作帧和 4139 个表情帧，训练窗口与真实音频时间发生倍数级错位。

旧实验 `beat_FM_aa_x0_aligned_v1` 虽然能在同样错位的验证集上取得较高 PCK/MSE，但其 checkpoint 不再具备音画同步有效性，也不能作为当前主模型或论文同步结论证据。其后续 SO(3) 边界处理记录仅保留为窗口拼接算法证据。

## 修复内容

1. 从 BVH `Frame Time` 解析真实帧率，将动作按统一时间戳从约 120 FPS 重采样到 15 FPS。
2. 将 60 FPS 面部系数按同一时间轴重采样到 15 FPS；音频保持 16 kHz。
3. 每个 split 写入内容哈希绑定的 `temporal_alignment_manifest.json`，记录源帧率、持续时间、公共前缀裁剪和目标帧数。
4. 运动 LMDB 升级为 v3，并绑定时间轴 manifest ID、15 FPS 和 16 kHz 契约。
5. HuBERT 缓存继续使用 v3，但重新按新 LMDB 的精确索引生成，并绑定运动 cache ID。
6. 缺失完整原始 WAV/JSON 的服务器副本只从旧缓存读取 16 kHz 音频和 60 FPS 面部源；动作始终重新读取原始 BVH，不复用旧动作帧。

关键实现位于 [preprocess_beat.py](../../preprocess_beat.py)、[datasets/beat.py](../../datasets/beat.py)、[cache_versions.py](../../utils/cache_versions.py) 和 [build_hubert_cache.py](../../build_hubert_cache.py)。回归测试位于 [temporal_alignment_checks.py](../../temporal_alignment_checks.py)。

## 全量源数据审计

| split | clip | 对齐后总帧数 | 记录公共前缀裁剪的 clip | 跳过 |
|---|---:|---:|---:|---:|
| train | 416 | 396,330 | 22 | 0 |
| val | 52 | 47,546 | 5 | 0 |

每个 clip 均满足动作帧数 = 表情帧数 = manifest 目标帧数，表情时间戳从 0 开始并严格以 `1/15` 秒递增，音频不短于对齐后的公共时间轴。完整机器审计见 [server_full_source_audit.json](../../bvh_output/temporal_alignment_smoke/server_full_source_audit.json)。

公共前缀策略只在音频与表情持续时间相互一致时允许裁剪 BVH 多余尾部；最大记录源尾部裁剪为 20.9637 秒，train 最大音频余量为 3.8 秒，均保留在 manifest 中供论文披露和后续异常样本复核。

## 缓存契约

| split | 运动样本数 | 时间轴 manifest ID | 运动 cache ID | HuBERT 形状 |
|---|---:|---|---|---|
| train | 38,468 | `ca16b939...594f1c` | `4680ae765e62451b9651d657b47d1acc` | `113 x 1024` |
| val | 4,609 | `9197df89...575001` | `4b95d723e2764e5bba7555b867f98ee7` | `113 x 1024` |

跨首、中、尾位置各抽检 17 个样本，姿态与轴角均为 `34 x 141`、表情为 `34 x 51`、音频为 36,266 samples，所有张量均为有限值。HuBERT 文件数与 LMDB 条目数完全一致，并绑定对应运动 cache ID。构建日志见 [运动缓存日志](../../logs/prepare_motion_cache_sync_v1.log) 与 [HuBERT 日志](../../logs/build_hubert_sync_v1.log)。

## 训练准入验证

先运行 2-epoch、每轮单 batch 的冒烟训练，checkpoint 中 552 个模型张量和 544 个优化器状态均无 NaN/Inf，且配置快照保存了两套运动/HuBERT cache 绑定。随后运行 5-epoch 全数据试跑，覆盖每轮 151 batches、共 755 个优化步：

- `final_loss`：首批区间约 1445.38，Epoch 4 末尾区间约 362.72。
- 验证：MSE `0.134254`、PCK `0.883288`、SRGR `0.876680`、Diversity `0.732054`。
- GPU：训练利用率约 91–100%，显存约 20.8 GB，无 OOM、NaN 或异常退出。

证据见 [冒烟日志](../../logs/train_beat_FM_aa_x0_aligned_sync_smoke_v1.log) 与 [5-epoch 全量试跑日志](../../logs/train_beat_FM_aa_x0_aligned_sync_pilot_v1.log)。这些数字只证明训练链路健康，不作为最终模型质量指标。

## 视觉与同步边界

[15 秒对齐 GT 冒烟视频](../../bvh_output/temporal_alignment_smoke/aligned_ground_truth_audio_smoke.mp4) 的音频与视频均从 0 开始且持续 15.000 秒；口型开合与音频包络的最佳相关滞后约 2 帧（0.133 秒），属于捕捉/音素级延迟量级，不再是倍数级时间轴错位。该视频是对齐后的 GT，不是 5-epoch 模型预测。

初版灰模视频只驱动 51 个 ARKit Shape Key，没有读取同一时间段的身体/头颈旋转，因此嘴部虽已同步，头部仍完全固定；同时将 Rocketbox 的整组 `f001_opacity` 材质隐藏，连眼睫毛与眼缘透明卡片也一并消失，视觉上像眼周破面。这两个问题都属于可视化管线，不是训练输出本身。

修复后的 [15 秒 GT 视频](../../bvh_output/temporal_alignment_smoke/aligned_ground_truth_clean_head_motion.mp4) 与 [左右前后对比](../../bvh_output/temporal_alignment_smoke/aligned_gt_face_render_before_after.mp4) 使用以下策略：

1. 从同一对齐 BVH 的 `Hips → Spine → Spine1 → Spine2 → Spine3 → Neck → Head` 链合成头部世界姿态，以片段首帧为相对零点后重定向到 Rocketbox `Bip01 Head`。
2. 可视化重定向增益为 0.7，安全上限为 22°；本片实际最大应用角约 12.99°、均值约 5.90°，未触发截断。
3. 将原 `f001_opacity` 的 28 个连通组件按空间区域拆分，仅保留 8 个眼/睫毛组件（208 个多边形）的原始 alpha 材质，其余头发卡片继续隐藏。
4. 表情系数不做额外时间平滑；头部姿态来自同源 GT 动作。视频仍是 GT 渲染，不代表 5-epoch 模型预测质量。

修复版共有 225 帧，视频 H.264 512×512/15 FPS、音频 AAC，两个流均为 15.000 秒。机器报告见 [头部姿态渲染报告](../../bvh_output/temporal_alignment_smoke/render_report_clean_head_motion.json) 与 [眼周材质拆分报告](../../bvh_output/face_model_validation/clean_clay_expression_smoke/report.json)。

再次逐帧复核后确认，剩余的“眼皮破损”不是眼皮拓扑撕裂，而是 `f001_head` 同时包含脸部和两个独立眼球。初版灰模替换整个头部材质时，虹膜、瞳孔和眼白也变成与眼皮相同的平灰材质，导致眼睑开口失去前后层次，看起来像空洞或破面。

最终灰模将 `f001_head` 的 3 个连通组件拆开：脸部主体继续使用纯灰材质，仅左右两个眼球组件恢复原始眼球纹理；两个眼球共 252 个多边形。高眨眼、最大睁眼、左右凝视和抬眼代表帧均已人工复核。最终 [15 秒完整眼部修复视频](../../bvh_output/temporal_alignment_smoke/aligned_ground_truth_complete_eyes.mp4) 与 [眼皮修复前后对比](../../bvh_output/temporal_alignment_smoke/eyelid_fix_before_after.mp4) 均保持原 GT 表情系数和头部姿态，不增加表情平滑。最终报告见 [完整眼部材质报告](../../bvh_output/face_model_validation/clean_clay_complete_eyes_smoke/report.json) 与 [最终渲染报告](../../bvh_output/temporal_alignment_smoke/render_report_complete_eyes.json)。

用户提供的眼部局部截图进一步暴露了上眼睑外侧的白色三角穿插。保留/隐藏睫毛卡片的 A/B 渲染确认，眼睑网格本身完整；伪影来自 Rocketbox 睫毛透明卡片。该卡片复用 `f001_opacity_color.tga` 的头发图集，ARKit 眼动变形后不能始终贴合眼皮。当前规范灰模因此保留两只纹理眼球，但隐藏全部透明睫毛卡片。最终 [无卡片 15 秒视频](../../bvh_output/temporal_alignment_smoke/aligned_ground_truth_final_clean_eyes.mp4) 和 [睫毛卡片移除前后对比](../../bvh_output/temporal_alignment_smoke/eyelid_card_removal_before_after.mp4) 中不再出现白色三角穿插；机器报告见 [最终无卡片渲染报告](../../bvh_output/temporal_alignment_smoke/render_report_final_clean_eyes.json) 与 [最终灰模材质报告](../../bvh_output/face_model_validation/clean_clay_final_no_cards_smoke/report.json)。

对用户随后提供的放大截图逐帧匹配后，仍可在第 79 帧复现一条与真实眼缘分离的深色上眼睑折痕。通道消融显示：清零眉毛、凝视、眯眼或面颊通道均不能消除该折痕，只有降低 `eyeWideLeft/Right` 才能消除；该帧原始 `eyeWide≈0.7544`。阈值 A/B 显示 Rocketbox 网格在 `0.3` 起出现明显分离折痕，`0.2` 仍保留连续眼睑。因此 `render_rocketbox_expression_video.py` 默认启用 `rocketbox_safe` 映射，仅对可视化中的两个 `eyeWide` 通道应用 `0.25×` 比例并限制到 `0.20`，保留逐帧变化而不把大多数帧饱和到同一阈值；原始 51 维 JSON、训练数据和指标均不改。可用 `--eye-safety-profile raw` 复现原始系数直驱结果。

修复后的 [15 秒眼睑安全视频](../../bvh_output/temporal_alignment_smoke/aligned_ground_truth_eyelid_safe.mp4)、[逐帧左右对比](../../bvh_output/temporal_alignment_smoke/eyelid_safe_before_after.mp4) 和 [第 79 帧放大对比](../../bvh_output/temporal_alignment_smoke/eyelid_safe_frame79_before_after.png) 均使用原纹理眼球且继续隐藏睫毛卡片。视频为 225 帧、512×512、15 FPS，音视频均为 15.000 秒；机器报告见 [眼睑安全渲染报告](../../bvh_output/temporal_alignment_smoke/render_report_eyelid_safe.json)。该片仍是对齐 GT 的可视化冒烟，不是正式重训模型预测；同一默认映射会自动用于后续预测流渲染。

## 下一步准入条件

- 用户已于 2026-08-30 明确确认启动 `beat_FM_aa_x0_aligned_sync_v1` 正式 500-epoch 实验。
- 正式训练必须复用本记录中的 manifest/cache ID，并在 checkpoint 配置中回读一致。
- 训练完成后重新生成 15 秒以上的 GT/预测/旧模型视觉对比，再评估表情—语音同步、动作自然度和默认后处理值。

## 正式训练启动记录

2026-08-30 20:46（Asia/Shanghai）完成服务器启动前检查：train/val 时间轴 manifest 均能通过内容哈希重算；运动缓存分别为 38,468/4,609 条，cache ID 为 `4680ae765e62451b9651d657b47d1acc` 与 `4b95d723e2764e5bba7555b867f98ee7`；HuBERT 文件数分别为 38,468/4,609，且 `source_motion_cache_id` 与对应运动缓存一致。`runner.py`、训练器、模型、数据集、选项与缓存契约等 7 个关键文件的本地/服务器 SHA-256 全部一致。数据盘剩余约 106 GB，48 GB GPU 健康。

启动时服务器仍在执行此前的 `fullval_blend7_v1` 有界全验证，约完成 2,220 个输出，GPU 占用约 2.9 GB。为避免并发抢占 GPU、污染训练吞吐和显存基线，没有中断该验证，也没有立即并发训练；服务器队列 PID `23262` 已设置为验证自然结束后自动使用锁定参数从零启动正式训练。队列日志为 `logs/queue_beat_FM_aa_x0_aligned_sync_v1.log`，正式日志为 `logs/train_beat_FM_aa_x0_aligned_sync_v1.log`。30 分钟自动监控 `diffsheg` 已切换为跟踪该队列与正式同步训练。
