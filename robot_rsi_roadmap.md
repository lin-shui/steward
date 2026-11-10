# Piper 孪生 RSI 路线图（PyRUA-REAL 模式落地）

> 2026-11-09 拍板。基于 Auriga / PyRUA-Lean / VISTA / PyRUA-REAL / RSIJudge 五个工作的分析，
> 把开发回路建在 Piper 孪生上，产物为冻结程序，真机做验证与物理差异收尾。
> 本文档是跨仓库工作文档，跟进 C0–C5 既有编号，C5 = sim2real。

## 1. 决策一句话

**参照 PyRUA-REAL 的模式（researcher/engineer 开发回路 + L0/L1/L2 经验库 + 冻结部署），但做两个改进：标定不留给 agent（ASPIRE 式固定流程），回路主体烧在孪生里（真机只做验证与收尾）。** 开发期用最强可用模型（一次性成本），部署期零模型调用；本地 DeepSeek 的角色是异常兜底与降本实验。

支撑数据（来源见 §8）：
- PyRUA-REAL：8 真机任务 91.1%，7/8 冻结部署；冻结比在线快 4.6–5×；思考占在线时间 63–90%。
- PyRUA-RSI：RoboDojo 固定程序 seed 0 82.8%，held-out seed 2 只掉 2.1 点 → 冻结产物是感知闭环技能，不是背布局。
- RSIJudge：回路代码全开源（`evolve/` + `learn/`，含回滚与 test 隔离）；CE 消融证明判据暴露 = 拟合验证器。

## 2. 总体架构（审查分级路由）

```
任务说明书(survey) ──▶ Researcher（方向/停止条件/经验注入）
                          │ experiment 单元
                          ▼
                      Engineer（写 cell、跑试验、诊断、修订）
                          │ cell = 唯一动作通道，网关在 cell 内
                          ▼
            robo API（Piper 孪生/真机同接口）
              action_preview() → action_execute(plan, tolerances) → receipt
              receipt.ok? 代码内调整重试（常态，零模型往返）
              receipt 异常 → 升级回 context 层（模型/人审，Ethernet 模式）
              感知(SAM3/几何) → Python 变量，context 只进决策增量
                          │
                          ▼
            技能库 library（skills + lessons，内容哈希 revisions）
              入库门：跨任务 Pareto + held-out 布局过关 + 局部伺服模式检查
```

分层原则：常态审查沉在代码层（PyRUA 效率），异常浮到 context 层（Auriga 可靠性），
成果固化到技能库层（VISTA/REAL 编译）。dsh 三档投递（pull/push/auto + 失败签名）= 路由器。

## 3. 组件映射

| 组件 | 现成品 | 缺口（要做的） |
| --- | --- | --- |
| 回路（experiment/回滚/晋升/test 隔离） | PyRUA-Lean RSIJudge 分支 `src/pyrualean/evolve/`(3.2k 行) + `learn/`(3k 行) | fork，learner 可先用单 agent |
| 技能库 | 同上 `evolve/library.py`（内容哈希、promotions.jsonl、candidate 静态校验） | 无 |
| L1 记忆 | dsh robot 插件 lessons 格式（与之同构） | 对齐卡片格式 |
| cell 沙盒 | PyRUA-Lean `sandbox.py`（AST 白名单） | 无 |
| robo API 模板 | `robodojo.py`（双臂，最接近 Piper）、`libero.py`（单臂+VLA） | **Piper RobotBase 子类（孪生+真机双 backend）** |
| 启动桥 | steward `packages/experimental/robot` robot_eval：SSH→conda→`pyrualean play --robot <name>` | 新 robot 名接入即用 |
| 感知 | spark SAM3 服务 + 4090 感知栈 | 接孪生/真机相机 |
| 标定 | ASPIRE `cap_harness/yam_real/calibration/`（ChArUco 手眼、15 位姿扫描、哈希 manifest；Apache-2.0 改编自 ENPIRE） | **改编到 Piper 工位（= C5 一部分）** |
| 安全 | — | **Piper 力限门（电流估计 or 腕 F/T，待硬件确认）** |

## 4. 四阶段计划

### 阶段一：孪生内跑通回路（~1–2 周）
- Piper `RobotBase` 子类接孪生 backend；fork RSIJudge `learn/campaign`。
- 首任务：抽屉（孪生里已完成、有失败余量）。
- 验收：冻结程序在孪生 held-out 布局过关（回滚与 test 隔离机制现成）。
- 方法论：沿用 `--context-file` 确定性注入做 A/B。

### 阶段二：C5 最小版（~1–2 周，标定主导）
- ASPIRE 管线改编到 Piper 工位 → L0 事实包（相机外参、桌面平面、关节极限、延迟）。
- 同一 robo API 切真机 backend，跑 3–5 真机验证局。
- 验收：孪生过关程序在真机的残差量化 → 决定阶段三真机时预算。

### 阶段三：真机小回路
- 只把孪生解决不了的物理差异放真机迭代（夹爪接触、摩擦、柔顺）。
- experiment 单元 + 聚焦子测试（不跑全任务）+ 人盯 + 力限。
- 验收：1–2 个任务族真机 80%+ 冻结部署。

### 阶段四：技能库与在线兜底
- 跨任务复用（dsh 入库门 + held-out 关卡）；Ethernet 式异常通道（DeepSeek 在线出场）。
- 降本实验：Flash 当 developer 需要几轮/多少监督（A/B 可测）。

## 5. 标定三件套（应对运行中环境变化）

1. **L0 固定标定**：ASPIRE 式离线管线 + 哈希 manifest（防文件被换）。
2. **L0.5 标定哨兵**：固定参照物（ArUco/标定块），每局开局 ~30 s 验证投影残差；
   超差即熔断停机上报，绝不带病运行（防物理被撞）。触发事件进 L1。
3. **任务层局部伺服**：精密技能默认"全局粗定位 + 腕部/接触局部闭环"，
   不赌全局标定；作为技能入库门的检查项。

场景层变化（物体/布局）一律由运行时重感知消化，不动标定——RSI seed 2 只掉 2.1 点即此分层的效果。

## 6. 红线（写进流程，不可协商）

1. **判据隔离（CE 教训）**：任务判据代码永远不进 learner 上下文；评测用独立忠实判据。
   RSIJudge 数据：判据暴露 51.6% vs 27.8% 的领先在忠实判据下消失（45.0% vs 47.2%）。
2. **未标定相机不出米制数据**（ASPIRE 边界原样搬）。
3. **技能入库三重门**：跨任务 Pareto（正负例都要看）+ held-out 布局过关 + 局部伺服模式。
   依据：记忆只在有失败余量的任务有效（dsh 实测 Long Swap +7 / Goal Task −11）；
   VISTA unseen composite 13.3% 是 per-suite 过拟合的下场。
4. **标定失效单独归类**：作为失败签名一类触发重标定例程，不触发任务层重试。
5. **真机异常分级**：力限熔断 > 人审 > 在线模型；冻结程序永不静默带病继续。
6. **真机新代码先沙盒**：复用 PyRUA-Lean AST 白名单；`sandbox_flags` 进审计记录。

## 7. 硬件清单（2026-11-09 部分确认）

本体 = 松灵 AgileX Cobot Magic 双 PiPER（agilex 分支命名来源）：

- [x] **腕部相机：有（Cobot Magic 标配双腕部相机）**。证据：官方 ALOHA 架构标配 +
  本仓 `third_party/piper_description/urdf/piper_description_with_camera*.xacro`
  即带腕部相机模型。→ PyRUA-REAL 式"全局粗定位 + 腕部局部伺服"策略可用，
  精密任务上限不降档；SAM3 感知接腕部相机流。
  待实物终验：相机型号/序列号清单（`ls /dev/video*` 或数采配置）。
- [ ] **腕部 F/T：大概率无**（PiPER 标配不含，URDF 无 ft/force link）。
  待实物终验：看夹爪与法兰之间有无带线圆柱铝块；查 AgileX 装箱单。
  若确认无 → 力限门改为**关节电流域**设计（piper_sdk 电流反馈 + 保守阈值 +
  电流突变检测），插接类任务的接触判据用"视觉 + 电流"混合；真要做
  Ethernet 级插接再后置选购六维力。
- [x] 移动底盘：有（C0–C4 已完成移动本体/跨工位差事，TRACER）。
- [x] 孪生环境进程：**spark**（2026-11-09 定）。且**孪生已在 spark**：
  `/home/lilin/game/uni-robot-agent/uni_robot_agent/integrations/cobot_magic/environment/`
  （`adapter.py` MuJoCo 后端 + `server.py`，与 `start_libero*.sh` 同一环境服务启动模式），
  阶段一 `piper_twin.py` 的 backend 即对接此 server。部署拓扑：spark = 孪生实例（多份并发）
  + 回路编排（`learn/campaign`）+ SAM3 服务 + 本地 DeepSeek，复用 robot_eval
  SSH→spark 通道；4090 = steward live 感知栈与真机在线兜底；Mac = 开发。
  并发规模待实测（按 spark 128GB 统一内存定每轮批大小）。
- [ ] 真机监督方式：初期人盯 + 工作区围栏的物理条件？（不阻塞阶段一/二）

## 8. 参考与本地代码位置

- PyRUA-REAL 论文 [arXiv:2611.06768](https://arxiv.org/abs/2611.06768)；项目页 dagroup-pku.github.io/PyRUA-Lean/real/。真机框架**未开源**。
- RSIJudge（回路开源）：PyRUA-Lean 仓库 `RSIJudge` 分支，arXiv:2611.08318。
- RSI 固定程序产物：PyRUA-Lean `main` 分支 `integrations/xpolicylab/PyRUA_RSI/programs/`（42 个，5.5 万行）。
- Auriga（执行层网关参照）：本机 `/Users/lilin/game/Auriga`，重点 `auriga/runtime/harness.py`、`AGENTS.md` 四条核心法则。
- VISTA（回路参照，选择协议反面教材）：本机 `/Users/lilin/game/VISTA`。
- ASPIRE 标定管线：本机 `/Users/lilin/game/ASPIRE/src/cap_harness/yam_real/calibration/`。
- steward 桥接：`packages/experimental/robot/`（robot_eval/robot_lesson，SSH→spark→rpent env）。

## 9. 阶段一开工清单（2026-11-09 已开工）

### 已完成（PyRUA-Lean 仓，分支 `piper-rsi`，基于 RSIJudge 分支）
1. 分支：`piper-rsi`（origin/RSIJudge 基底）。**未合并 main**：两支历史不相关；
   RSIJudge 自包含（play/robots/hosts/evolve/learn 齐全），直接在其上开发。
2. `src/pyrualean/piper_twin.py`：`PiperTwinRobot(RobotBase)`，双臂 grasp-site 位姿、
   pitch/yaw/roll 与孪生四元数互转（z=approach，待烟雾测试确认）、`move_to` 含
   `preview=True`（goal-regions 求解不移动）、`set_gripper`/`home`/`locate`/`segment`/
   `back_project`，`ADAPTER` 已注册。
3. `src/pyrualean/hosts/piper_twin.py`：每 episode 启动孪生 server
   （`python -m uni_robot_agent.integrations.cobot_magic.environment.server`），
   stdlib HTTP 客户端（`/info`、`/captures`、`/steps`、`/goal-regions`、`/records`、
   `/artifacts`、`/finish`，版本化 envelope），动作按 ≤5cm/0.3rad 增量循环，
   每步落盘三视角 PNG + 标定。
4. `src/pyrualean/hosts/piper_sight.py`：RGB-D 感知（SAM3 mask → 深度投影 →
   支撑面/中心/顶高/外廓/长轴/grasp_yaw）。
5. `src/pyrualean/knowledge/piper_twin.md`：模型可见操作知识（不写死桌面高度）。
6. 测试：`tests/test_piper_twin.py` 11 项通过；`test_robots`/`test_prompt` 通过；
   全量 280 passed（1 个既有 `cv2` 依赖缺失，与本改动无关）。

### spark 侧（2026-11-09 已实跑，冒烟全绿）
- [x] 一条龙脚本（PyRUA-Lean `scripts/`）：`sync_to_spark.sh`（含 uni-robot-agent）、
  `spark_env_setup.sh`（含 `mcp>=1.8,<2`）、`piper_twin_smoke.py`、`spark_bringup.sh`；
  入口 `SPARK_HOST=spark-direct bash scripts/spark_bringup.sh`。
- [x] episode 解释器 `/home/lilin/env/piper-twin`：`uni_robot_agent` + `steward` + `mujoco`
  同环境；数据根 `/home/lilin/game/steward/var/steward`。
- [x] 直连链路 `spark-direct`（169.254.10.2，user liaoheng）同步 + 运行；Wi-Fi 仍是默认路由。
- [x] 冒烟 **12/12**：下探 7.1 mm / 1.6° / pitch 89.0°，home 0.011 rad / 23 步；
  `--with-perception` 点提示 locate：center (0.4996, 0.1984, 0.7944)、top 0.8169、
  height 0.045。
- [x] host 补 `card`/`run` 子命令与 `run_episode`/`card_only`/`Cell.suite`；
  `pyrualean card|run --robot piper_twin ...` 可用；`play_episode` 直通。
- [x] 手写 policy 跑通（completed，5 calls / 536 steps / 3 相机 / result.json）。
- [x] **SAM3 服务**：checkpoint 在
  `/home/liaoheng/robodojo/src/XPolicyLab/policy/PyRUA_Lean/checkpoints/sam3.pt`
  （3.3GB，sha256 `9999e234…`；之前 find 少一层漏了）。`scripts/start_sam3_server.sh`
  用 RPent conda env 起 HTTP 服务 `127.0.0.1:8114`，policy 侧 `SAM3_ENDPOINT` +
  `sam3_local.RemoteSam3`（不需要 torch/sam3）。**合成渲染上文本提示 0.02-0.14，
  点提示 0.97**：`locate/segment` 支持 `point=[row,col]`，知识文件已写明。
- [x] `piper_sight._support_z` 改为 mask 附近 0.25m 内 z 中位数：原全图 5% 分位把
  头相机看到的地面当支撑面（height 1.29m），现在 bowl 的 height 0.045m。
- [x] 首个模型驱动 v0（本地 `pyrua-coder-4b`，llama-server :8899，131k ctx）：
  第一轮 10 cells / 2112 calls / 20000 steps 耗尽；接 SAM3 后重跑 21 cells / 153 calls /
  12308 steps / 600s 超时。模型能测到 plate (0.534,0.416,0.772) 和 blue object，但
  **碗/盘身份弄反**，把平盘当碗抓（holding False），随后代码反复 TypeError。
  失败性质 = 感知可用、语义识别错，正是回路可学的余量。
- [x] 抽屉任务：`scripts/make_piper_drawer_scenario.py` 生成 `piper-drawer` 场景 +
  scenario（柜子 x=1.2、yaw=π；机器人复用带三相机的 `cobot-magic` 资产）；adapter 加
  `drawer-open` 判据（`drawer_slide` q ≥ 0.10 m，已注入 qpos 验证 true/false）；
  card/run 通过。
- [x] **worker = dsh + deepseek-flash（Mac）**：`scripts/dsh_worker.py` 把 Claude Code 的
  `-p/--output-format stream-json|json` 表面转成 `dsh --profile headless --json -`，
  转发事件并产出 campaign 解析的 result；key 从 `DEEPSEEK_API_KEY` 或
  `~/.dsh/.credentials.yaml` 读取。`check_services` 实测 policy + worker 均通过。
- [x] **Mac 成为 campaign 主机**：`/Users/lilin/env/piper-twin`（py3.12、mujoco 4.0.0、
  steward/uni/pyrualean、mcp<2）；host 的 `start_twin` 按平台选 `MUJOCO_GL`
  （darwin=cgl，`PIPER_TWIN_GL` 覆盖）；Mac 冒烟 12/12；SAM3 走
  `ssh -f -N -L 8114:127.0.0.1:8114 spark-direct` 隧道；策略模型走
  `http://169.254.10.2:8899/v1`。
- [x] **piper campaign 分支**：`learn/config.py` 的 `benchmark: "piper"`（scenarios 与
  layouts 对齐、dev/test 场景不相交、twin_python/twin_root/max_episode_steps）；
  `episode.py` 的 `_piper_episode` + 场景哈希 identity；`configs/learn/piper/drawer.json`。
  **library 任务名必须 `[A-Za-z0-9_]+`**：campaign task `drawer_open`，`missions` 映射
  孪生 mission `drawer-open`。
- [x] **4 个 drawer 布局**：`make_piper_drawer_scenario.py --cabinet-x/--cabinet-y`；
  dev `piper-drawer-a/b`（1.16,-0.10 / 1.24,0.10），test `piper-drawer-c/d`
  （1.18,0.12 / 1.22,-0.12）。
- [x] **v0 基线（rounds:1）**：`campaign init/run/report` 跑通；dev 0/2，test 0/2
  （v0=v0），4 episodes / 0.36h / ~1M input tokens（85% cached）。失败模式：l0 两次
  **overclaim**（模型 finish success，孪生 `task_success=False`），l1 dev 3 cells
  烧完 20000 steps，l1 test 600 s 超时。
- [x] **第一个真正的 RSI 轮次**（round 2）：piper 版 `PIPER_BRIEF` 与 replay 已实现
  （`worker.py` 的 piper 分支、`scoring_sources=[]`、blind 模式隐藏 twin data root；
  `learn.replay` 无需改即可在孪生跑 `cell_*.py`，实测 22.7 s/局）。dsh + deepseek-flash
  worker 11 分钟完成：读 2 个 dev episode、写
  `candidate/tasks/drawer_open/skills/drawer.py`（白色把手 + 蓝色柜体回退、抓后校验
  `holding`、同姿态拉 0.36 m、用抽屉前板位移 ≥0.08 m 验证）+ `lessons/drawer.md`，
  跑 3 个 replay，FINDINGS 精确指出 l0 的 overclaim 和 l1 的点提示打到墙上。accept 生成
  revision `2845e466f77db2ab`。结果：dev 0/2，test 0/2 vs v0 0/2（无提升、无回退）；
  policy 在 l1 第 438 s 才尝试 skill，600 s 超时。worker 自己的 replay 也没有成功抓取，
  skill 是几何推理而非端到端验证。
- [x] **技能修复（手动，round 1 后）**：四个 layout 全部由孪生判定成功。三处根因：
  1) **场景快照过期**：`piper-drawer-*` 的场景里 robot 实体是旧版资产，lift 关节
     origin x=0.0，而 `CobotIK`/当前 `cobot_magic()` 是 0.24。适配器每条指令都比
     真实手臂多想 0.24 m（IK 与 sim 的 FK 差恰好 0.2399 m），所以“到点”全是假的。
     `make_piper_drawer_scenario.py` 现在直接从当前 `cobot_magic()` 发布
     `cobot-magic` 资产（补三相机），重建 a–d。
  2) **腕部极限**：lift 0.60 时 down 姿态的 wrist 比肩高 5.6 cm，笛卡尔增量从折叠
     HOME 起步解不出（目标 wrist z≈1.6 m）。改用 `move_joints` 到标定 pregrasp
     （lift 0.70，pinch 在把手上方 0.112 m、工具朝下），再三段短 `move_to` 下降。
  3) **姿态漂移**：host 每步从测量姿态 slerp，配合 `max_tilt=1.0` 会让工具角漂
     30–100°；现在一旦进入 0.3 rad 内就保持目标四元数，孪生 `--max-tilt 0.2`。
  另外把手在 +y 侧优先用左臂（b 的右臂在够取极限）。复现命令：
  `make_piper_drawer_scenario.py --lift 0.70` + 4 个 `pyrualean run` 直接 replay
  `open_drawer`（a/b/c/d 全 success）。新 revision `b1304e69d1e362ae`（parent
  `2845e466f77db2ab`），lesson 标 `dev_verified`。
- [x] **用最终 revision `7c6a703dd827ef0d` 跑 dev+test（4B policy）**：dev a/b 均
  `success=True`（2 cells / 65 s、2 cells / 62 s，claim agreed），held-out test c/d 均
  `success=True`（5 cells / 115 s、4 cells / 179 s）。策略确实 `from library.drawer import
  open_drawer` 并直接完成，不再 wandering。过程中修掉两个 infra bug：
  ① 4B 服务中途掉线（`model provider failed mid-episode`）触发 infra retry，
  `max_launches` 20 被吃满 → 提到 40；② test d 首次跑遇到 `409 sequence_mismatch`
  （丢回复导致客户端序号永久落后，后续每个动作都被拒，policy 烧完 30 cells）；
  已在 `Client.post` 加自愈：409 且 `next_sequence` 可读时对齐序号重发一次。修后 d
  一次成功。注意 `campaign report` 仍显示 final=2845e（手工冻结/激活未走 round 账本），
  原始证据在 `~/game/piper-campaign/episodes.jsonl`（campaign 全量已从 /tmp 迁至 `~/game/piper-campaign/`，含 runs/library/videos/debug）。
- [x] **所谓“4B 服务掉线”的根因与修复**：不是服务不稳（llama-server PID 自 11-06
  未重启）。模型写了近 2 万字符的 cell，在 `max_tokens=4096` 处被截断成半截 JSON
  tool call；harness 把这条坏 assistant 消息留在 transcript 里，下一次请求把它重放给
  llama-server，`--jinja` 模板解析 `arguments` 时抛 500（日志
  `Failed to parse tool call arguments as JSON ... missing closing quote`，响应仅
  3–11 ms），重试原样重发同一份历史 → 必死，被记成 `provider error mid-episode`。
  修复：① `openai_runtime.py` 在任何 `_parse_arguments` 失败时把该调用的
  `arguments` 替换成 `"{}"` 再存入历史，且 `finish_reason == "length"` 时把错误换成
  “被输出上限截断，请写更短的 cell”；② `max_tokens` 4096 → **8192**（4B 实测
  50–55 tok/s，8192≈150 s/次，8192 是 600 s policy 预算下的合理上限；12288≈225 s
  需同时抬 `policy_seconds`，16384≈300 s 不建议）；③ 知识文件加“cell 保持几十行”。
  回归测试 `test_a_truncated_tool_call_is_never_replayed_as_invalid_json`；24 个
  runtime 测试全过。修后 test d 再跑一局 success（11 cells/380 s，未再出现 500）。
- [ ] **下一关**：① 预算/延迟：600 s policy 对 4B 仍紧（d 首次跑到 596 s）；`max_cells`
  30 与 `policy_seconds` 600 的配比可再调。② 本地推理或更强 API 模型做对照。③
  `slots.per_gpu=1` 串行，test 4 局要 13 分钟；并发/预算要调。④ worker replay 预算（3）
  仍应优先验证完整抓–拉。

### 实跑中发现并修掉的两个真 bug
1. uni adapter `_joint_limits` 用裸名 `L_joint1` 查 MuJoCo，而孪生命名是
   `w:cobot-1::L_joint1`；`mj_name2id` 返回 -1 后读到最后一个 joint 的 `[0, 0]`，
   **所有 joint target 都被拒**（home 在关节空间全废）。改为经 `sim.build.names`
   解析 + 后缀扫描兜底。
2. `Gripper.holding` 原判据是“闭合且 opening > 0.01”，空夹时 0.012 误报 True；
   改用孪生 `metadata.gripper_holding`（空夹 False，实测通过）。

### 预算口径（实测）
`max_episode_steps` 计的是 **sim tick**：一次 settle 最多 400 tick，一次 60 增量的
长移动约 4000 tick。孪生任务预算要 20000 量级，不能按 control step 估。

### 验收对照（阶段一结束）
抽屉任务冻结程序 vs dsh 记忆版 vs 裸 cell 版三方 A/B（`--context-file` 注入法）。
