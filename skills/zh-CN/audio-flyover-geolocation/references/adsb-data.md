# ADS-B 数据：来源、格式、基准与缺口

## 目录
1. 数据源与下载
2. readsb trace 格式
3. OpenSky 与通用 CSV
4. 高度基准（最常见的精度杀手）
5. 机型分类
6. 覆盖缺口与法规
7. 实操建议

---

## 1. 数据源与下载

| 来源 | 覆盖/粒度 | 获取 | 备注 |
|---|---|---|---|
| **adsb.lol globe_history** | 全球，每天一个 release；每架飞机一个 `trace_full_<hex>.json` | GitHub `adsblol/globe_history_YYYY` releases（大文件切分为 `.tar.aa`、`.tar.ab`…，先 `cat` 合并再解压） | 免费，ODbL 许可；首选 |
| **ADS-B Exchange 样本** | 全球，**仅每月 1 日** | `https://samples.adsbexchange.com/traces/YYYY/MM/01/<xx>/trace_full_<hex>.json`（xx = hex 末两位）；另有 `readsb-hist`（5 s 快照）与 hires-traces | 样本免费；其他日期需商业授权，注意其使用条款 |
| **OpenSky Network** | 全球（接收站覆盖为准），状态向量约 1 Hz（存档时间分辨率因年份而异） | Trino（`trino.opensky-network.org`，表 `minio.osky.state_vectors_data4`，按 `hour` 分区）；Python：`pyopensky` / `traffic` | 需要账号并申请研究访问权限 |
| 自建 readsb / tar1090 | 本地站点 | `/var/globe_history/YYYY/MM/DD/traces/` | 低空覆盖最好 |

只下载**日期 + 时间窗 ±1 h + 区域外扩 ~0.5°** 所需的数据。adsb.lol 单日全球归档为 GB 级，
解压后用 `adsb-ingest --bbox` 过滤即可（脚本按 `trace_full_*` 递归 glob，自动识别 gzip）。

OpenSky Trino 查询示例（注意 `hour` 分区条件，否则极慢）：
```sql
SELECT time, icao24, lat, lon, velocity, heading, vertrate, baroaltitude, geoaltitude, onground
FROM minio.osky.state_vectors_data4
WHERE hour BETWEEN 1789221600 AND 1789239600          -- 整点 unix 秒
  AND time BETWEEN 1789223000 AND 1789238000
  AND lat BETWEEN 38.5 AND 39.7 AND lon BETWEEN -95.4 AND -93.8
```
导出 CSV 后 `adsb-ingest --format opensky --src export.csv`。

## 2. readsb trace 格式

文件顶层：`icao`（hex）、`r`（注册号）、`t`（ICAO 机型代码）、`desc`、`timestamp`（基准 unix 秒）、`trace`（数组）。
每个点（wiedehopf/readsb `README-json.md`）：

```
[dt, lat, lon, alt_baro_ft | "ground" | null, gs_kt, track_deg, flags, vrate_fpm,
 details | null, source, alt_geom_ft, geom_rate_fpm, ias_kt, roll_deg]
flags: 1 = stale, 2 = 新航段开始, 4 = vrate 为几何, 8 = 高度为几何
```
脚本处理：t = timestamp + dt；遇到 flags&2 或间隔 > `--max-gap`（45 s）切段；丢弃地面点；
速度用报文中的 gs/track/vrate（比差分位置平滑得多），缺失时才用 Savitzky-Golay 差分。

## 3. OpenSky 与通用 CSV

- `--format opensky`：列 time, icao24, lat, lon, velocity(m/s), heading, vertrate(m/s), baroaltitude(m), geoaltitude(m), onground。
- `--format generic`：`hex,t,lat,lon,alt_m,gs_ms,track_deg[,vr_ms,type]`，alt_m 视为 MSL。

## 4. 高度基准

斜距 d 对高度误差几乎 1:1 敏感（飞机在头顶附近时）。

- **气压高度（alt_baro）**：相对标准大气 1013.25 hPa 的压力高度，不是真实高度。
  真实 ≈ baro + (QNH − 1013.25) × 8.3 m（另有温度误差，冷天真实高度更低）。
  QNH 取录音时刻最近机场的 METAR（如 Iowa State Mesonet ASOS 存档），传 `--qnh`。
- **几何高度（alt_geom / geoaltitude）**：GNSS 高度，WGS84 椭球高（HAE）。MSL = HAE − N。
  N 为大地水准面差距（美国本土约 −20 ~ −35 m，即 MSL 比 HAE 高 20–35 m；中国东部约 −5 ~ +5 m 不等，
  可用 GeographicLib GeoidEval 查 EGM2008）。传 `--geoid-n`。
- DEM（Copernicus GLO-30）高程参考 EGM2008（≈ MSL），因此飞机高度要换算到 MSL 后再与 DEM 相减。
- `--alt-source auto`：有几何高度用几何高度，否则用 QNH 修正后的气压高度。

## 5. 机型分类

用 ICAO Doc 8643 描述码（如 L1P = 陆上单发活塞、L2T = 双发涡桨、L2J = 双发喷气、H1T = 单发涡轴直升机）：
第 1 位 L/S/A/H/G/T（陆机/水机/两栖/直升机/旋翼机/倾转旋翼），第 3 位 P/T/J/E（活塞/涡桨-涡轴/喷气/电动）。
脚本读取 tar1090-db 的 `icao_aircraft_types.js`（缓存于 `~/.cache/afg`，离线时用内置简表），并用 ADS-B
emitter category（A1–A7、B1–B7）补充：
`prop`（L*P、L*T）、`rotor`（H*、G*、A7）、`jet`、`glider`（B1）、`balloon`（B2）、`unknown`。
搜索默认事件类 `prop,rotor,unknown`；喷气机只参与静默判定。

## 6. 覆盖缺口与法规

- 美国 14 CFR 91.225：A 类、B/C 类空域及其上方、Mode C veil（30 nm）内、10,000 ft MSL 以上（2,500 ft AGL 以下除外）
  等需要 ADS-B Out；**这些区域之外、低于 10,000 ft 的 GA 不强制**——低空乡村地区的小飞机可能完全不在数据里。
- 军机、执法、部分 HEMS 直升机可能关闭或使用匿名/隐私 ICAO 地址（PIA）；FAA LADD 列表飞机在部分平台被屏蔽，
  但 adsb.lol / ADSBx 通常不过滤。
- 接收覆盖：低空（< 1,000 ft AGL）远离接收站时会断续或缺失；山区更甚。航迹插值跨越 > 45 s 的空白不可信。
- 欧洲：EASA 对 5.7 t 以上或 250 kt 以上的飞机要求 ADS-B（Mode S EHS/ELS），GA 常只有 Mode S 无位置。
- 结论："数据中无匹配" ≠ "不在该区域"。报告时写明覆盖假设。

## 7. 实操建议

1. 先用宽 bbox + 时间窗 ingest，看 `tracks.npz` 中 prop/rotor 航班数量；太多时再缩小。
2. 检查目标时段的 prop 航迹是否在低空有连续数据（`plot` 不了就用 Python 画一下）。
3. 时间窗要包含 tCPA ± 传播延迟 ± 时钟不确定度，再加 10 分钟余量。
