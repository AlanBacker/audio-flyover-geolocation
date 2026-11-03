# 来源

## 方法来源（colsto）
- colsto, *I discovered a sound that leaks your exact location (and it's everywhere)*, YouTube, 2026-11-02 —
  <https://www.youtube.com/watch?v=UytYrhYLOKI>（完整录音：<https://youtu.be/4CxVxuMVkMc>）。
  说明栏方法摘要：ADS-B + ICAO 机型分类；200 m 网格；3D 斜距 + DEM；CPA 与传播延迟；
  频谱约束 v 与 CPA 斜距；920 s 偏移 + 静默约束；发射时刻径向速度精确建模；锁定双事件打分；
  已知时间的 1 分钟片段做全美搜索仍排第一；讨论高通、掩蔽、通话压缩为何无效。作者声明不发布代码——
  本 skill 的实现为独立重写。
- colsto, *Can you geolocate a video using background thunder sounds?*, 2026-08-09 — <https://www.youtube.com/watch?v=SnbkIm56TfI>
- colsto, *I located a fan's video using thunder sounds*, 2026-09-30 — <https://www.youtube.com/watch?v=6a7Z0I3LAnE>

## 报道与讨论
- TechRadar: *YouTuber discovers aircraft noise trick that can geolocate your videos to within 100m* —
  <https://www.techradar.com/computing/cyber-security/youtuber-discovers-aircraft-noise-trick-that-can-geolocate-your-videos-to-within-100m-doppler-effect-pinpoints-locations-from-just-60-seconds-of-audio>
- Simon Willison: *A sound that leaks your exact location* (2026-11-03) — <https://simonwillison.net/2026/Nov/3/a-sound-that-leaks-your-exact-location/>

## 学术背景
- B. G. Ferguson, "A ground-based narrow-band passive acoustic technique for estimating the altitude and speed
  of a propeller-driven aircraft," *J. Acoust. Soc. Am.* 92(3), 1403–1407 (1992)——由单麦克风窄带 Doppler 曲线
  估计螺旋桨飞机速度、高度与 CPA 时间的经典方法（本 skill 的 v/d/f0 恒等式与其一致）。

## 数据与格式
- readsb JSON / trace 格式：<https://github.com/wiedehopf/readsb/blob/dev/README-json.md>
- 机型表（ICAO Doc 8643 描述码）：<https://github.com/wiedehopf/tar1090-db>（`db/icao_aircraft_types.js`）；
  ICAO Doc 8643：<https://www.icao.int/publications/doc-8643-aircraft-type-designators>
- adsb.lol 历史数据：<https://github.com/adsblol/globe_history_2026>（ODbL）
- ADS-B Exchange 样本数据：<https://www.adsbexchange.com/data-products/sample-data/>（`samples.adsbexchange.com`，每月 1 日）
- OpenSky Network Trino：`trino.opensky-network.org`（需研究访问权限；Python 客户端 `pyopensky`、`traffic`）
- Copernicus DEM GLO-30（AWS 公共桶）：<https://copernicus-dem-30m.s3.amazonaws.com/>

## 法规
- 14 CFR 91.225（ADS-B Out 要求的空域）：<https://www.law.cornell.edu/cfr/text/14/91.225>
