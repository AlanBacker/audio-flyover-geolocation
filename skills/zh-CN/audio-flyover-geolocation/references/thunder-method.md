# 雷声 × 闪电数据定位（模式 T）

colsto 早先两期视频的方法（2026-08《Can you geolocate a video using background thunder sounds?》、
2026-09《I located a fan's video using thunder sounds》），精度为公里级，适合没有飞机曲线但有雷暴的录音。

## 原理

每次闪电在时刻 t_i、位置 x_i 产生雷声；观测点 p 在 t_i + |x_i − p|/c 听到，幅度随距离衰减。
对候选点 p 和录音起始时间 T0，可合成"应听到的雷声包络"，与录音的低频（20–250 Hz）RMS 包络比对：
- 斜距：闪电源高度取 ~2 km（`--source-h`），r = √(水平距离² + h²)。
- 幅度模型：A ∝ 1/r^p（`--p` 1 = 幅度、2 = 能量），r 下限 1.5 km（避免近处闪电独占）、上限 25 km（更远基本听不到），并加指数衰减。
- 相似度：余弦相似度；对所有 T0 偏移用 FFT 互相关一次算完，并做窗口能量归一化。
- 网格默认 2 km；在 [`--start`, `--end`] 内搜索录音起始时间。

## 数据

| 来源 | 位置精度 | 获取 |
|---|---|---|
| Blitzortung / LightningMaps | ~km 级（视频中约 3 km 偏差） | 社区网络；历史数据需自行记录实时 feed 或申请；遵守其使用条款 |
| MET Norway（挪威气象局）闪电 API | ~百米级（视频中约 200 m） | `api.met.no`（需 User-Agent 标识），覆盖北欧 |
| 各国气象局 / 商业网络（NLDN、GLD360、ENGLN 等） | 百米级 | 多为商业授权 |

整理为 CSV：`time,lat,lon`（time 为 ISO 或 unix 秒）。

## 使用

```bash
$AFG thunder --audio storm.mp4 --strikes strikes.csv --bbox 59.5,60.5,10.0,11.5 \
    --start 2026-09-20T15:00Z --end 2026-09-20T17:00Z --out work/thunder.json
```
输出按相似度排序的候选点与对应 T0。

## 限制与建议

- 精度受闪电定位误差、声源高度（云闪/地闪）、地形遮挡、风和逆温影响——公里级是常态。
- 偏远的孤立闪电（主雷暴团之外）信息量最大：它们只在少数位置"恰好"对齐。视频第二期即对其加权。
- 录音中的风噪、雨声、说话会污染 20–250 Hz 包络；先剪出雷声明显的片段。
- 结果作为区域先验，可再与飞机曲线（如果有）或视觉线索结合。
