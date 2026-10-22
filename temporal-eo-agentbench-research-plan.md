# Temporal-EO-AgentBench：从时序遥感指令 Benchmark 到可验证 Tool-Trace Agent
github 同步：https://github.com/hiiCasper/TEO-Agent.git
## 概述

TEOChat 将时序遥感任务组织成 instruction benchmark；本研究进一步将时序遥感任务组织成 executable agent benchmark。也就是说，评估目标不只是不看最终答案是否正确，还要评估 agent 是否选择了正确工具、是否产生了有效中间证据、是否定位了正确的时间变化，并能否通过后训练学会可验证的时序遥感工作流。

本工作的定位应当避开“又一个 EO agent”或“又一个 VLM benchmark”，重点强调：

> Temporal EO + Tool Trace + Agent Post-training + Evidence-grounded Evaluation

相关工作中，TEOChat 主要面向 temporal EO instruction tuning；ChangeGPT 已使用 Agent 组织变化分析工具，但其论文没有报告对 Agent controller 进行 tool-trace SFT 或偏好训练；OpenEarthAgent 已经使用经过验证的多步工具轨迹训练通用 EO Agent；Earth-Agent、OpenEarth-Agent、ThinkGeo 等也已覆盖工具调用、轨迹评估或开放环境工作流。因此，本研究不能再把“训练一个会调用遥感工具的 Agent”或“同时评估过程与结果”单独作为新颖性主张。本研究应把核心增量收紧为：**面向遥感变化任务的专用可训练 Agent，以及利用变化任务内在时间关系对答案、证据与工具轨迹进行跨执行监督的技术方法**。

参考工作：

- TEOChat: https://arxiv.org/abs/2410.06234
- ChangeGPT: https://arxiv.org/abs/2601.02757
- OpenEarthAgent: https://arxiv.org/abs/2602.17665
- OpenEarth-Agent: https://arxiv.org/abs/2603.22148
- Earth-Agent: https://arxiv.org/abs/2509.23141
- ThinkGeo: https://arxiv.org/abs/2505.23752
- ESA-EO-Bench: https://arxiv.org/abs/2502.09481
- RSure-Agent: https://arxiv.org/abs/2610.04836
- ChangeStar / temporal symmetry: https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Change_Is_Everywhere_Single-Temporal_Supervised_Object_Change_Detection_in_Remote_ICCV_2021_paper.html
- Metamorphic Testing（1998）: https://www.cse.ust.hk/faculty/scc/publ/CS98-01-metamorphictesting.pdf

## 核心想法

构建一个新的 benchmark，暂名为 **Temporal-EO-AgentBench**。每个样本包含以下核心字段：

- `task_instruction`：自然语言时序遥感任务，例如灾害损毁评估、城市扩张、建筑变化、水体变化、植被或作物变化。
- `temporal_inputs`：agent 可见的任务输入。该字段的内部结构不固定，应根据具体任务和原始数据集决定。它可以包含图像标识符、时间戳、传感器类型、可用波段、感兴趣区域、空间范围或数据集提供的上下文信息。需要 agent 通过工具自行探索的信息不应出现在这里。
- `output_schema`：agent 可见的输出格式定义，用于约束 agent 最终回答。该字段应避免包含 `summary`、`confidence` 等自然语言字段或与本研究无关的字段，以便自动评估。
- `gold_answer`：标准答案，格式必须与 `output_schema` 对齐。
- `gold_tool_trace`：专家标注或半自动生成的可执行工具调用轨迹，包括数据检查、裁剪、配准或对齐检查、云或无效像素掩膜、变化检测、分割、统计和验证等步骤。


## 核心贡献

1. **最主要贡献——面向变化任务的专用可训练 Agent**：提出一个以 7B 级文本模型为 controller、以遥感变化工具为外部能力的 change-centric Agent，并在经过真实执行与证据核验的变化任务 tool trace 上进行后训练。与 ChangeGPT 的主要区别是，ChangeGPT 更接近由闭源 LLM 在推理时进行规划和工具编排的系统，其论文没有报告对 Agent controller 进行 tool-trace SFT、DPO/RL 或权重更新；本研究显式训练 controller 学会变化任务中的工具选择、依赖组织、证据读取和错误修复。但必须承认，OpenEarthAgent 已经训练了通用 EO tool-use Agent，并包含 change-detection 输入，因此不能把“第一个经过 tool-trace 训练的 EO Agent”作为论文主张。更稳妥的定位是：**针对变化检测和时序变化理解，训练一个能够产生、核验并修复变化证据链的专用 Agent**。
2. **重要技术贡献——Temporal Metamorphic Process Supervision（TMPS）**：提出一种面向时序遥感 Agent 的跨执行过程监督方法。它把变化任务天然具有的时间反演、无变化恒等、空间协变和配准扰动关系编译为可执行的关系 oracle，同时检查原始任务与变式任务之间的最终答案、空间证据、统计量和工具轨迹是否按预期保持不变或发生确定性反演；随后定位第一个违反关系的工具步骤或 artifact，并把修复动作与原错误动作构造成局部偏好数据，用于 SFT/DPO。该方法的目的不是要求 Agent 复现唯一 gold trace，而是利用变化任务的结构规律判断多条候选路径是否可执行、证据是否忠实、时间方向是否正确。
3. **数据贡献——带 tool trace 与时序关系的 benchmark 和标注流程**：构建带有可执行 tool trace 的变化任务 benchmark，并设计从原始图像对、问题、答案和 mask 出发，生成专家/半自动轨迹、证据 artifact、时序变式样本、关系规格和局部偏好对的可复现标注流程。训练样本仍保持最小公开字段；来源、变式关系、provenance 和审计信息保存在私有 manifest 或构建元数据中。

## 重要技术贡献：Temporal Metamorphic Process Supervision（TMPS）

### 1. 为什么需要 TMPS

单次任务中“最终答案正确”并不能证明 Agent 真正使用了双时相影像和中间证据。Agent 可能依赖答案先验、问题模板或单期图像猜中答案；变化检测工具也可能因为错配准、亮度、季节或传感器差异产生伪变化，而 Agent 将一个正常执行但内容错误的 observation 继续传递到最终答案。当前 RSRCC 中严重的 `yes/B` 答案先验进一步说明，只报告 Accuracy 无法证明时序遥感推理能力。

TMPS 的基本思想是：不只判断一次执行是否正确，而是对同一任务施加具有明确语义的时序干预，得到一个或多个变式任务，然后检查多次执行之间是否满足变化任务必然具有的关系。设一次 Agent 执行为：

```text
Fθ(x) = (answer, evidence_artifacts, tool_trace)
```

其中 `x=(I_t1, I_t2, question)`。对输入施加变换 `g` 后，TMPS 不要求两次输出逐字相同，而要求：

```text
Fθ(g(x)) 与 ρg(Fθ(x)) 满足任务定义的关系
```

`ρg` 分别规定答案、证据和工具过程在该变换下应该保持不变、随空间坐标协变，或按照变化方向发生反演。

### 2. 变化任务中的关系类型

| 时序干预 | 最终答案应满足的关系 | 证据应满足的关系 | 工具过程应满足的关系 |
| --- | --- | --- | --- |
| 交换 `t1/t2` 并同步改写问题 | “是否变化”等无向答案保持不变；`increase ↔ decrease`、`constructed ↔ demolished`、`appeared ↔ disappeared`、`from ↔ to`；有符号差值取反 | 无向 change mask 指向同一变化区域；from/to 语义交换 | before/after 参数、对象匹配方向、事件推断方向和差值计算顺序必须同步反转 |
| 构造 `(I_t1,I_t1)` 或 `(I_t2,I_t2)` | 对适用的纯变化问题应为 `no`、0 或空结果 | change mask 为空 | 不应依靠问题模板或多数类先验制造变化 |
| 对两期共同施加受控亮度、颜色或几何变换 | 变化语义保持不变 | mask 对共同几何变换协变，恢复到参考坐标后应一致 | 工具路由和结论应对与变化无关的共同成像扰动保持稳定 |
| 只对一期注入受控平移、旋转或局部错配准 | 配准修复后的语义答案应与原任务一致 | 证据需映射回统一参考坐标 | Agent 应发现配准异常并在变化检测前调用检查/校正工具，不能把边缘错位当作真实变化 |
| MCQ 选项重排 | 正确选项的语义不变，输出字母按新位置改变 | 证据不变 | 防止依赖 `A/B/C/D` 位置先验 |

并非所有关系都适用于所有问题。系统必须根据 `question_type` 和变化事件 ontology 使用白名单规则。例如，二值 change mask 在时间交换后通常具有对称性，但“新增/拆除”“增加/减少”和 `from/to` 具有方向性，不能错误地要求所有输出保持不变。无法确定变换后语义的样本不得自动进入训练。

### 3. Relation Compiler：生成变式任务和关系规格

`Relation Compiler` 是一个以确定性规则为主的变式任务编译器。它读取原始问题类型、答案 schema、可用证据和变化事件 ontology，输出：

1. 变换后的时序输入和问题；
2. 两次执行之间的 `answer_relation`；
3. mask、polygon、对象集合和统计量的 `evidence_relation`；
4. 工具参数、关键步骤和 artifact 依赖的 `process_relation`。

例如，原题为“第一期到第二期是否新建了建筑”，交换时相后应改写为“是否拆除了建筑”，并产生如下内部关系规格：

```json
{
  "transformation": "temporal_reverse",
  "answer_relation": "preserve_yes_no_and_invert_event",
  "evidence_relation": "change_region_invariant",
  "process_relation": [
    "swap_temporal_arguments",
    "reverse_object_correspondence",
    "invert_directional_event"
  ]
}
```

该关系规格属于 benchmark 构建和训练时的内部 manifest，不要求加入用户已经确定的五字段最小样本 JSON。

### 4. Cross-execution Verifier：跨执行检查答案、证据和过程

Agent 分别在原始任务和变式任务上真实执行工具。`Cross-execution Verifier` 不单独评价某条文本轨迹是否“像 gold”，而是根据 Relation Compiler 给出的关系同时比较：

- `answer_relation_score`：答案是否保持、反转或映射到正确语义；
- `evidence_relation_score`：mask、polygon、对象集合、计数和面积等 artifact 是否满足不变、反演或空间协变关系；
- `process_relation_score`：关键步骤、时相参数、artifact 依赖和配准前置条件是否正确；
- `cost_score`：是否存在无意义重复调用或明显不必要的高成本操作。

概念上可写为：

```text
R_meta =
    λa · answer_relation_score
  + λe · evidence_relation_score
  + λp · process_relation_score
  - λc · unnecessary_tool_cost
```

该加权式只用于表达评价维度，不能让高答案分数抵消确定性的证据或过程违规。对于时相参数颠倒、配准前置条件遗漏、空变化任务产生非空 mask 等可程序判定的硬关系，应先进行 hard gating；只有通过硬约束的候选轨迹，才使用连续分数比较证据质量、成本和多条合法路径。

过程评价使用关键步骤和依赖关系，不使用唯一工具序列 exact match。两条工具顺序不同但均可执行、证据充分且满足时序关系的轨迹，都应被视为合法路径。

### 5. Relation-Violation Localization：定位第一个关系违反节点

每次工具调用产生的 raster、mask、vector、table、render 和 report 均以 artifact ID 进入依赖图。TMPS 对齐原始执行与变式执行中的语义等价 artifact，并找到第一个不再满足关系的节点。例如：

```text
单边平移后仍错误地判定 registration_status=pass
→ 变化 mask 出现大量建筑边缘伪变化
→ 建筑计数错误
→ 最终答案错误
```

尽管最终存在多个错误，训练时应优先监督第一个违反关系的节点——配准检查或其后的决策，而不是只把整条失败轨迹作为负样本。对于无法由自动规则可靠定位的错误，应标记为 `unresolved` 并进入人工审计，不强行生成偏好对。

### 6. 从关系违反生成 SFT 与局部偏好数据

在首个违反关系的状态保留此前已经验证的正确前缀和真实 observation，然后产生并实际执行多个候选动作：

```text
正确前缀 + 当前 observation
├── chosen：check_coregistration → register_images → detect_change
├── rejected-1：直接 detect_change
├── rejected-2：调用正确工具但时相参数相反
└── rejected-3：证据不足时直接输出答案
```

只有真实执行后能够修复关系违反、产生合法 artifact 并完成任务的动作才能标为 `chosen`。训练数据分为两类：

- **合法替代轨迹**：与 gold trace 不同，但答案正确、证据充分、可执行且满足全部时序关系，作为新增 SFT 数据；
- **局部偏好对**：在相同正确前缀下，能够修复首错的动作作为 `chosen`，原始错误或证据质量较差的动作作为 `rejected`，用于 DPO。

TMPS 因此形成以下闭环：

```text
原始任务
→ Relation Compiler 生成变式任务和关系规格
→ Agent 对原始/变式任务分别 rollout
→ Cross-execution Verifier 比较答案、证据和过程
→ Relation-Violation Localization 找到首个违反节点
→ 实际执行反事实候选分支
→ Alternative-path SFT + Seed-SFT Replay + Local Counterfactual DPO
```

该闭环可以直接嵌入后文的三轮训练策略。第一轮仍进行 seed tool-trace SFT；第二、三轮则优先从新样本的跨执行关系违反中生成新增 SFT 数据和局部 DPO 偏好对。

### 7. 与相关工作的边界

- **不是首次提出 temporal symmetry**：ChangeStar、ChangeMask 等变化检测模型已经用时间交换对称性约束 change feature 或二值 mask。TMPS 的增量是把题型相关的 invariant、equivariant 和 directional relation 提升到 Agent 的自然语言答案、证据 artifact 与工具过程层。
- **不是首次训练 EO tool-use Agent**：OpenEarthAgent 已在经过验证的结构化工具轨迹上进行 SFT。TMPS 的区别是利用同一任务的多次关系化执行产生过程 oracle 和后训练信号，而不是只验证单次轨迹。
- **不是首次验证遥感 Agent 的中间 observation**：RSure-Agent 已要求工具返回过程证据，并在推理时结合当前证据与历史可靠率决定接受、补证或拒绝。TMPS 必须突出其不同点：任务变换生成、跨执行关系验证、时间方向建模、关系违反定位以及向 SFT/DPO 数据的回流。
- **不是把通用 metamorphic testing 直接改名**：Metamorphic Testing 自 1998 年起已用于软件测试、科学计算、机器学习和自动驾驶。TMPS 的研究问题是如何把遥感变化语义编译为可执行关系，并用这些关系训练而不只是测试 tool-use Agent。

因此，论文不应声称“首次使用时间交换”或“首次验证遥感 Agent”。在完成更系统的全文、代码和同期论文检索前，建议使用以下受限表述：

> We introduce a change-centric cross-execution process-supervision framework that compiles temporal remote-sensing relations into executable constraints over answers, evidence artifacts, and tool trajectories, and converts relation violations into localized post-training signals.

### 8. 数据适配与实验要求

QAG-360K 同时具有文本答案、问题相关 mask 和 `CN/CtW/CfW/IN/DN/LC/SC/CR` 八类题型，更适合作为 TMPS 的像素证据和题型关系主实验。RSRCC 可用于对象事件、时间方向、空间指代和 MCQ 先验实验，但其原生数据没有 mask，且当前本地快照存在严重答案不均衡，因此不能单独承担“证据忠实性得到提升”的主要结论。

至少报告以下新增指标：

- `Temporal Relation Satisfaction Rate`：时间变换后满足答案关系的比例；
- `Directional Event Accuracy`：新增/拆除、增加/减少、from/to 等方向事件的正确率；
- `Evidence Equivariance IoU/F1`：证据恢复到统一参考坐标后的空间一致性；
- `Pseudo-change False Positive Rate`：同图比较、共同辐射扰动和单边错配准下的伪变化率；
- `Registration-trigger Recall`：存在错配准时主动检查并修复的比例；
- `First Relation-violation Localization Accuracy`：首个关系违反节点定位准确率；
- 原有 answer、executability、tool argument、critical-step coverage 与 cost 指标。

核心消融应至少包括：

```text
Trace-SFT
+ ordinary outcome DPO
+ answer-only temporal relation
+ answer + evidence relation
+ answer + evidence + process relation（完整 TMPS）
```

还应分别移除 temporal reversal、no-change identity、registration intervention、artifact-level verifier 和首错局部偏好，以证明改进来自时序关系过程监督，而不是简单增加数据量或执行次数。

## Benchmark 设计

该 benchmark 可以从 TEOChat 类时序任务扩展而来，但需要将纯文本答案升级为结构化输出和可执行工作流。

### 已登记的数据来源与核验状态

以下数据集计划作为 Temporal-EO-AgentBench 的候选原始样本来源。RSRCC 当前本地快照已经完成初步字段、答案分布、文件完整性和 split 泄漏审计，结果见下文“RSRCC 本地数据质量审计”；许可、上游版本差异和人工图文一致性仍待核验。QAG-360K 在正式纳入前仍需核验数据字段、许可、访问方式、训练/测试划分，以及是否能够支持可执行的时序工具轨迹标注。

| 数据集 | 官方来源 | 拟定 Benchmark 作用 |
| --- | --- | --- |
| RSRCC | [Google / Hugging Face：RSRCC](https://huggingface.co/datasets/google/RSRCC) | 为双时相遥感变化理解任务提供图像对及其自然语言描述，拟用于构造变化描述、变化区域/类型判定等时序任务，并补充对应的 `gold_tool_trace`。 同一个图像对会对应多条不同问题。| 
| QAG-360K | [VisTA 项目仓库](https://github.com/like413/VisTA) | 为遥感视觉问答与地理空间问答任务提供候选样本，拟筛选或重构其中具有多时相、变化或时间定位语义的样本，用于构造结构化问答和工具调用轨迹。 |

> 注意：上述数据集是 Benchmark 的**来源**，而不是直接等同于最终 Benchmark。需要将原始样本统一转换为 `task_instruction`、`temporal_inputs`、`output_schema`、`gold_answer` 与 `gold_tool_trace`，并在测试集划分中避免与模型训练数据泄漏。

### 来源数据集的题型

| 数据集 | 原生题目/答案形式 | Benchmark 中的直接评测价值 |
| --- | --- | --- |
| **RSRCC** | 基于双时相影像的两类问答：（1）**Yes/No 二元问答**，如是否新建建筑；（2）**四选一选择题（A–D）**，如某建筑、道路或植被发生了什么变化。题目还包含整体场景变化程度判断。 | 可直接构成二元 Accuracy 与 MCQ Accuracy 子集。 |
| **QAG-360K** | CDQAG 三元组：**自然语言问题 + 文本答案 + 对应变化区域 mask**，而非原生 A–D 选择题。其 8 类“8C”问题为：是否变化（CN）、变成什么（CtW）、由什么变来（CfW）、是否增加（IN）、是否减少（DN）、变化最大类别（LC）、变化最小类别（SC）、变化比例（CR）。其中 CN/IN/DN 是判断类；CtW/CfW/LC/SC 通常回答土地覆盖或语义类别；CR 为数值或比例区间。 | 既可评测答案，也可使用 mask 评测变化区域定位与答案的证据一致性。 |

题型来源：[RSRCC dataset card](https://huggingface.co/datasets/google/RSRCC)、[RSRCC 样例](https://huggingface.co/datasets/google/RSRCC)、[QAG-360K / VisTA 论文](https://arxiv.org/html/2410.23828)、[QAG-360K 官方仓库](https://github.com/like413/VisTA)。


### 来源数据集的格式
RSRCC:
{
  "before": 图像_t1,
  "after":  图像_t2,
  "text":   "Question: ...\nAnswer: ..."
}

QAG-360K:
{
  "image_t1": 图像_t1,
  "image_t2": 图像_t2,
  "question": 问题,
  "answer": 文本答案/类别答案,
  "mask": 与此问题答案对应的二值像素掩膜，是一个二值二维数组
}

### RSRCC 本地数据质量审计（2026-10-21）

以下结论来自当前本地 `RSRCC/` 副本的逐行 metadata 检查、文件存在性检查、文件头检查和跨 split 图像内容 SHA-256 比对。它们描述的是**当前本地快照**，不能在未核对上游版本的情况下外推到所有 RSRCC 发布版本。审计区分“已确认问题”和“尚待人工视觉核验的问题”；模型或工具答错本身不能作为原始 gold 错误的证据。

#### 1. 本地 split 概况

| Split | metadata 记录 | 唯一图像对 | 图像文件 | 当前可用性 |
| --- | ---: | ---: | ---: | --- |
| Train | 87,016 | 6,417 | 12,834 | 1 条标注无法由当前解析器解析；其余 87,015 条可解析 |
| Validation | 无本地 metadata | 266 | 532 | 只有图像，没有问题和答案，不能直接用于模型选择 |
| Test | 21,999 | 1,826 | 3,652 | 全部可由宽松解析器解析，但存在格式与内容污染 |

已确认的正常项：Train/Test metadata 引用的图像文件均存在，before/after 文件名在各条记录内具有一致的 pair stem，没有发现缺文件或显式文件名错配。但文件名一致不等于问题语义与图像一定对应；后者仍需要人工视觉审计。

#### 2. 答案分布严重失衡

训练集答案分布：

| 题型 | 答案 | 数量 | 题型内比例 |
| --- | --- | ---: | ---: |
| Binary | `yes` | 43,299 | 97.70% |
| Binary | `no` | 1,019 | 2.30% |
| Multiple choice | `A` | 17,382 | 40.71% |
| Multiple choice | `B` | 23,194 | 54.32% |
| Multiple choice | `C` | 2,076 | 4.86% |
| Multiple choice | `D` | 45 | 0.11% |

测试集答案分布：

| 题型 | 答案 | 数量 | 题型内比例 |
| --- | --- | ---: | ---: |
| Binary | `yes` | 10,753 | 96.60% |
| Binary | `no` | 378 | 3.40% |
| Multiple choice | `A` | 4,459 | 41.03% |
| Multiple choice | `B` | 5,885 | 54.15% |
| Multiple choice | `C` | 511 | 4.70% |
| Multiple choice | `D` | 13 | 0.12% |

因此，仅按题型输出多数答案——Binary 恒答 `yes`、MCQ 恒答 `B`——即可在训练集已解析记录上达到约 **76.42%**，在测试集达到约 **75.63%**。这会产生严重的答案先验捷径：高 Accuracy 不再能证明 agent 使用了图像、工具或变化证据。此前固定随机种子抽取的 100 条测试样本中，Binary gold 为 47 个 `yes`、1 个 `no`；按题型多数类作答可得 74/100，而文本工具 agent 得到 75/100。因此该 75% 不能作为视觉变化推理有效性的证据。

后续必须同时报告：多数类 baseline、balanced accuracy、macro-F1、各答案类别召回率、按题型分层的结果，以及 evidence/process 指标。训练时可使用类别均衡采样或 loss weighting；MCQ 可在保持选项语义和 gold 不变的前提下重新随机排列 A–D 位置。自然分布测试集与平衡诊断测试集应分开保存和报告，不能只保留一个人为平衡后的测试集。

#### 3. 标注格式与生成内容污染

“能够被宽松解析器读取”不等于原始标注干净。当前本地训练集有 260 条记录不符合严格的规范答案结尾，其中 214 条没有使用统一的 `**Answer:**` 标记；测试集对应数量分别为 61 和 49。部分记录只是 Markdown 写法不同，但也存在明确的生成污染。

训练集中至少 11 条记录混入 `Wait`、`Note`、自我纠错、重复题目、重新排列选项或不确定性文本，CSV 行号为：`6924, 23967, 27430, 29725, 37580, 40662, 42620, 51311, 56026, 66717, 74304`。其中至少以下 4 条在中途重写了问题或选项，并复用了新的选项字母；当前解析器会把最终字母映射到最初的错误选项语义：

- 第 42,620 行：最初 `A=pruned`，后续重写为 `A=demolished`；
- 第 51,311 行：最初 `A=planted`，后续重写为 `A=removed`；
- 第 56,026 行：最初 `A=grown`，后续重写为 `A=demolished`；
- 第 74,304 行：最初 `A=road removed`，后续重写为 `A=new road constructed`。

训练集第 65,585 行缺少可解析的标准 Question 标记，是当前唯一直接解析失败的记录。第 24,848 行与第 71,004 行是完全相同的重复记录。

测试集中至少 7 条记录明显混入生成过程文本或额外数据，CSV 行号为：`1451, 6748, 8845, 9138, 10256, 20367, 21500`。其中：

- 第 9,138 行先给出 `Answer: A`，随后明确说明 A 错、B 才正确，并以 `Corrected Answer: B` 结束；当前解析器会取 A；
- 第 20,367 行答案字段是一段解释，末尾明确说正确答案是 B，但文本以 `A talent...` 开头；当前解析器会误取 A；
- 其他记录包含 `IP [DONE]`、分析过程、重复 Question/Answer、经纬度或自然语言解释。

这些异常记录必须进入 quarantine，不得直接转换为 `gold_answer` 或 `gold_tool_trace`。清洗程序应保留 `raw_text`，只接受唯一且规范的最终答案；一旦出现多个问题、多个答案、`Corrected Answer`、选项重排或答案解释，必须人工裁决，不能依靠“取第一个字母”或“取最后一个字母”的规则自动解决。

#### 4. Train/Test 图像内容泄漏

按图像内容 SHA-256 检查，发现 5 个完整 before/after 图像对同时出现在 Train 和 Test。它们使用不同 UUID 文件名，但两期图像内容逐字节相同：

| Train pair stem | Test pair stem | Train 问题数 | Test 问题数 |
| --- | --- | ---: | ---: |
| `2b2d00e1_5c67_48f0_a530_1963d1217123` | `4e259227_3bc6_4096_85b6_40f26b6b5047` | 2 | 2 |
| `a9b41619_650d_42c1_b150_7e95c7d7022c` | `64c89d8c_32ce_48a2_96d8_d1d9685e3f26` | 2 | 2 |
| `360c8197_d181_476a_b722_e793787a5758` | `47112980_a942_4e40_8d2c_2f1abb8e2c26` | 49 | 37 |
| `9ce47c3f_6360_4e68_896a_5cb097e8d3e4` | `4382c527_267b_4fe5_abbb_0aa76433dd70` | 12 | 10 |
| `c966bdba_a58f_471c_8b3c_715ffe837132` | `92b86ead_b1b4_4011_9738_4dff34bafc5b` | 2 | 2 |

泄漏共涉及 67 条训练问题和 53 条测试问题。两侧没有发现完全相同的整段 question/answer 文本，但图像场景已经泄漏，因此 Test 不再是严格独立的视觉评估。此前随机 100 条测试没有命中这些泄漏 pair，但完整 benchmark 必须重新划分。

正式数据构建必须先按 before/after 内容哈希建立 canonical pair ID，再按 pair 分组划分 Train/Val/Test；UUID、文件名和问题文本都不能作为唯一去重依据。还应增加近重复图像检测，以发现重新压缩、裁剪或轻微色彩变化后的同源图像。

#### 5. 图像扩展名与真实编码不一致

当前本地 Train 的 12,834、Validation 的 532、Test 的 3,652 个图像文件均以 `.png` 命名，但文件头全部为 JPEG，即总计 17,018 个“PNG 文件名、JPEG 内容”的文件。Pillow 等内容感知解码器可以读取，但这会影响按扩展名选择解码器、格式统计、数据校验和对外发布。

构建 Benchmark 时必须二选一：保持原始字节并将文件扩展名改为 `.jpg`，同时记录原始文件名；或者无损解码后重新编码为真正的 PNG，并记录转换脚本与 checksum。不得只修改后缀而不修改内容。

#### 6. 图像对权重不均与模板复用

训练集中每个图像对对应的问题数差异很大：最少 1 题、最多 125 题、中位数 6 题、均值 13.56 题。如果按问题记录均匀采样，少数场景会被重复训练几十到上百次，导致场景记忆和 pair 权重失衡。此外，有 337 种完全相同的 annotation text 被用于多个图像对，主要集中在全局一致性问题，表明数据具有明显模板化特征；模板复用本身不一定是错误，但可能让模型依靠问题措辞预测答案。

训练 sampler 应先采样 image pair，再在该 pair 的问题中采样；或者设置每个 pair 每 epoch 的最大问题数。数据划分、SFT 三轮划分和评测都必须以 canonical image pair 为最小隔离单位。还应报告每个 pair 的问题数分布、模板重复率和按 pair 宏平均的性能。

#### 7. 尚未确认的图文与 gold 正确性问题

此前随机 100 题的执行审计确认了文件存在、输入可读、trace 无 gold 泄漏和输出格式合法，但没有对 100 对图像逐题进行人工视觉审标。因此：

- 没有在该 100 题中确认图文完全不对应或 gold 明确错误；
- 也不能据此声称这些标注全部正确；
- “tree was demolished”等措辞不符合自然对象事件表达，应规范为 `removed/cut down`，但针对抽查样本，图像仍可能支持“树被移除”的核心语义；
- “high visual parity/scene consistency”类全局问题在结构不变但季节、亮度或植被外观变化明显时存在判定标准歧义，必须明确它指结构一致还是像素/外观一致。

正式纳入 Benchmark 前，应按题型、对象类别、答案类别、问题模板、变化幅度和 split 分层进行人工视觉抽审，并重点复核少数类 `no/C/D`、全局一致性题、复杂空间指代题和所有自动清洗异常。人工审计应记录 `verified_correct`、`annotation_error`、`ambiguous`、`image_text_mismatch` 与裁决说明。

#### 8. RSRCC 数据准入门槛

RSRCC 只能作为候选来源数据，不能将当前原始 metadata 直接视为 Benchmark gold。进入 7B controller 的 Tool-trace SFT、DPO 或最终评测前，必须完成：

1. 冻结本地原始快照，记录来源、版本、文件清单和 SHA-256；
2. 使用严格 schema 解析 Question、options、answer，异常记录全部 quarantine；
3. 人工裁决解析失败、多答案、自我纠错、选项重排和疑似错误 gold；
4. 以图像内容哈希和近重复检测重新构造 pair-level Train/Val/Test，消除 5 组已知泄漏；
5. 获取原始 Validation 标注，或从清洗后的 Train 按 canonical pair 重新划分开发集；
6. 处理 JPEG 内容与 `.png` 后缀不一致，并保存可复现转换记录；
7. 对 Binary 和 MCQ 答案先验进行平衡采样/选项重排，并保留自然分布评测用于外部可比性；
8. 使用 pair-aware sampler，限制单一 image pair 的训练权重；
9. 完成分层人工视觉审计后，才将样本标为 `gold_answer` 和 `gold_tool_trace`；未人工确认的自动轨迹只能标为 `silver_trace`；
10. 在 Dataset Card 中公开清洗规则、移除数量、答案分布、pair 分布、泄漏检查、人工一致率和 majority baseline。

在这些门槛完成前，可以使用当前 RSRCC 副本调试工具接口、artifact 流转和 trace schema，但不能用其原始 Accuracy 证明模型具有时序遥感推理能力，也不应直接扩展为大规模正式训练数据。


### 问题格式统一策略

本 Benchmark **不改变原始问题所问的变化语义，也不将 QAG-360K 的开放类别题强行改成 A–D 选择题**；在构建样本时，只将不同来源的题目包装为统一的任务接口，并显式给出答案格式约束。agent 在训练中学习在完成工具调用后按约定格式作答。

```text
Given a before image and an after image, answer the question using the available tools.
Question: {raw_question}
Output format: {answer_format}
Return only the answer in the specified format.
```

| 统一题型 | 适用来源题型 | `answer_format` 与规范输出示例 |
| --- | --- | --- |
| `binary` | RSRCC Yes/No；QAG CN / IN / DN | `one lowercase token: yes or no` → `yes` |
| `multiple_choice` | RSRCC A–D | `one uppercase option letter: A, B, C, or D` → `A` |
| `semantic_class` | QAG CtW / CfW / LC / SC | `one canonical lowercase land-cover class` → `water` |
| `change_ratio` | QAG CR | `percentage or percentage range, e.g. 15% or 10%-20%` → `10%-20%` |

每条样本保留 `raw_question`、`raw_answer` 与来源标识，以保证可追溯；同时新增 `question_type`、`answer_format` 和规范化后的 `gold_answer`。评测分别报告格式合法率、格式合法条件下的答案准确率，以及二者同时满足的严格准确率；QAG-360K 另报告基于其 gold mask 的定位/证据分数。

第一版建议优先选择四类任务，保证故事线清楚、工具链明确：

- **Disaster damage assessment**：xBD 类任务。工具链包括灾前/灾后对齐、建筑检测、损毁分类和区域统计。
- **Building change detection**：S2Looking / LEVIR-CD 类任务。工具链包括双时相配准检查、变化检测、mask-to-polygon 转换和变化计数。
- **Urban development monitoring**：QFabric 类任务。工具链包括多时相区域裁剪、状态分类和时间定位。
- **Land / water / vegetation change QA**：Sentinel / Landsat 指数类任务。重点体现遥感指数、云掩膜和多时相变化分析。

样本本身不需要 `difficulty` 字段。可以在数据划分或评测配置层面组织三种评估模式：

- **Answer-only evaluation**：只评估最终结构化答案。
- **Evidence-grounded evaluation**：要求最终答案能够被正确时间步和区域证据支持。
- **Workflow-level evaluation**：评估工具调用、中间结果和验证行为。

## 端到端数据构建与训练流程

本研究的完整流程不是先生成一批文本轨迹再直接训练，而是按照“来源冻结与清洗 → pair-level 数据隔离 → 工具实现与验收 → canonical trace 构建 → TMPS 变式与关系标注 → 三轮训练 → 独立评测”的顺序执行。所有自动生成内容都必须经过真实工具执行和 verifier 检查；LLM API 或 Codex 只作为候选问题、候选轨迹和候选修复动作的提出者，不能直接充当 gold 标注器。

```text
RSRCC / QAG-360K 原始数据
→ 来源登记、许可核验、快照冻结和数据清洗
→ canonical image-pair 去重与 Train / Val / Test 划分
→ Train 内部再按 pair 划分 Seed 60% / Round-2 20% / Round-3 20%
→ 实现并验收覆盖目标题型的工具库
→ LLM/Codex 提议 canonical tool trace
→ ToolExecutor 真实执行 + artifact/verifier 检查 + 人工抽审
→ Relation Compiler 生成适用的 TMPS 变式任务和 relation_spec
→ 原始任务与变式任务真实执行，形成可审计的关系化轨迹
→ Seed Trace-SFT 得到 Agent_0
→ Round-2 rollout、逐错修复、Alternative-path SFT + Seed-SFT Replay + Local DPO 得到 Agent_1
→ Round-3 rollout、逐错修复、Alternative-path SFT + Seed-SFT Replay + Local DPO 得到 Agent_2
→ Val 选择超参数和 checkpoint，Test 只进行最终一次评测
```

### 阶段 1：来源冻结、数据清洗与 pair-level 划分

1. 下载并冻结 RSRCC 与 QAG-360K 的原始快照，记录来源、版本、许可、文件清单和 checksum；原始数据只读保存，后续处理均写入派生目录。
2. 按前文的数据准入规则处理解析异常、重复记录、图像编码不一致、疑似错误标注和答案先验；无法自动裁决的记录进入 quarantine。
3. 先根据 before/after 图像内容建立 `canonical_pair_id`，再按 image pair、事件和近重复组划分 Train / Val / Test。不得先按问题行随机划分。
4. 在 Train 内部继续按 canonical pair 划分 `Seed-SFT 60% / Round-2 20% / Round-3 20%`。同一原始图像对产生的所有问题、轨迹和 TMPS 变式必须继承同一个 split 和训练轮次，防止原始样本与变式样本跨集合泄漏。
5. Val 只用于阈值、采样比例、loss weight、replay 比例和 checkpoint 选择；Test 不参与轨迹生成筛选、偏好构造或任何梯度更新。

### 阶段 2：工具实现、能力验收与版本冻结

根据最终选入数据的实际题型和词汇审计工具覆盖，而不是从通用遥感任务列表反推工具。每个工具必须具有明确的输入/输出 schema、artifact 类型、失败状态、模型或算法版本、预处理参数和验收测试。工具完成单元测试、契约测试和代表性开发样本验收后，冻结 tool registry、checkpoint checksum、阈值与运行环境；否则后续生成的 trace 无法复现。

### 阶段 3：canonical tool trace 构建

对清洗后的原始任务，使用自动化脚本结合 LLM API 或 Codex 提议一个或多个候选工具轨迹，但标注流程必须是：

```text
LLM/Codex 提议候选 trace
→ 参数与 schema 静态检查
→ ToolExecutor 真实执行
→ 检查 artifact 依赖、答案、证据、成本和输出格式
→ 对疑难、冲突和低置信度样本进行人工裁决
→ gold_trace / silver_trace / rejected_trace / quarantined_trace
```

- `gold_trace`：真实执行成功，答案和证据正确，关键过程经过规则检查及必要的人工确认；
- `silver_trace`：真实执行成功并通过自动 verifier，但尚未完成人工确认，只能用于方法开发或经过单独控制的训练实验；
- `rejected_trace`：不可执行、证据不支持答案、违反硬约束或使用错误时相/参数；
- `quarantined_trace`：源标注、工具 observation 或 verifier 之间存在无法自动解决的冲突。

`gold_tool_trace` 是一条经过确认的参考解法，而不是唯一合法解法。后续过程评价和训练不得用完整序列 exact match 排斥其他可执行且证据充分的路径。

### 阶段 4：TMPS 变式数据构建

Relation Compiler 根据 `question_type`、答案 schema、事件 ontology 和可用证据，为每条基础样本选择适用的 TMPS 关系。不是每条数据都机械生成所有变式；语义无法确定或无法构造可靠 oracle 的变换必须跳过。

每个变式至少保存以下内部信息：

```json
{
  "base_sample_ref": "...",
  "transformation": "temporal_reverse | identity_pair | common_nuisance | misregistration | option_permutation",
  "transformed_task_ref": "...",
  "relation_spec": {
    "answer_relation": "...",
    "evidence_relation": "...",
    "process_relation": ["..."]
  },
  "provenance": {
    "compiler_version": "...",
    "tool_registry_version": "...",
    "verification_status": "..."
  }
}
```

LLM/Codex 可以协助改写变式问题或提出变式 trace，但确定性的时间方向、答案映射、mask 关系和必要工具前置条件必须由 Relation Compiler 与 verifier 决定。原始任务和变式任务均需真实执行；满足关系的可执行轨迹可作为 SFT 候选，关系违反轨迹则作为后续错误定位和偏好构造的来源。

### 阶段 5：三轮训练与独立评测

第一轮只使用 Seed 子集建立基本工具能力；第二、三轮分别在此前不可见的训练图像对上进行 rollout、关系验证和逐错修复。每次定位并修复一个关系违反后，都从修复后的真实环境状态继续 rollout；若后续出现新错误，则继续生成新的局部偏好对，直到当前任务成功结束、达到调用预算或进入无法可靠裁决的 `unresolved` 状态。

第二、三轮的新增 SFT 不能完全替代第一轮数据。两轮 SFT 都必须混入一部分经过确认的 Seed-SFT 样本进行 replay，以维持工具调用格式、schema 遵循、基础工具选择和标准工作流；具体组成和更新顺序见后文“三轮训练策略”。完成训练后，Val 用于模型选择，Test 同时报告自然分布结果和 TMPS 压力测试结果，但不得把 Test 上发现的错误回流到训练。

## Agent 主实验

agent 使用统一工具库。第一版不应把重点放在动态“造工具”上，而应强调 agent 是否能正确使用和验证时序遥感工具。

最小工具库可以包括：

- metadata / projection inspector
- temporal image loader and cropper
- co-registration / alignment checker
- cloud or invalid-pixel mask
- change detector / differencing / segmentation model
- object detector or building extractor
- zonal statistics / area calculator
- visual evidence renderer
- consistency verifier

训练路线：

- **SFT**：使用 `gold_tool_trace` 训练 agent 生成合理工具序列。
- **Process Preference**：构造好/坏 trace 对，偏好更少遗漏关键步骤、证据正确、成本更低且可执行的轨迹。
- **Verifier-guided Post-training**：使用自动 verifier 对中间证据、最终答案和工具调用合法性打分，再优化 agent。

Baselines：

- **Direct VLM**：TEOChat / Qwen2-VL / GPT-4o 类模型直接回答。
- **Generic tool-use agent**：没有遥感专门 trace 训练的通用工具调用 agent。
- **Specialist pipeline**：每类任务的传统专家模型或专家流程。
- **Ablated agents**：去掉 post-training、verifier 或时序遥感机制的模型变体。

## 数据规模与模型训练设置

第一版 benchmark 计划包含 **10,000** 条时序 EO agent 样本，划分如下：

```yaml
Train: 7,000
Val: 1,000
Test: 2,000
Total: 10,000
```

该规模适合作为高质量 agent benchmark 的第一版。由于每条样本不仅包含最终答案，还包含结构化输出约束和多步 `gold_tool_trace`，单条样本的信息密度高于普通遥感 VQA 或 caption 数据。对于 7B 级文本 LLM，7,000 条高质量 tool-trace 样本足以支撑领域 SFT 和后续偏好训练。这里的 7,000 条必须来自完成 pair-level 去重、split 重划分、异常标注隔离和人工抽审后的干净样本；当前原始 RSRCC 记录和未经人工确认的自动 trace 不计入该规模。

任务类型应围绕 EO 时序变化任务展开，优先覆盖：

- **灾害评估**：建筑损毁、受灾范围、灾前/灾后对比。
- **建筑变化检测**：新增建筑、拆除建筑、建筑区域变化。
- **水体变化检测**：洪水扩张、湖泊或河道变化、水体面积变化。
- **陆地覆盖变化**：裸地、城市用地、耕地或其他 land-cover 转换。
- **植被变化检测**：植被退化、作物长势变化、NDVI / EVI 等指数变化。

模型规模计划最大使用 **7B 文本 LLM**。该模型不直接承担底层遥感视觉识别任务，而是作为 tool-use planner / workflow controller，负责理解任务、解析 `temporal_inputs`、选择工具、组织工具调用顺序、读取工具 observation，并按 `output_schema` 输出结构化答案。

### 三轮训练策略

训练过程分为三轮。全部比例均相对于 Train 计算，而不是相对于包含 Val/Test 的 10,000 条数据计算；对于当前 7,000 条训练数据，三部分分别为 4,200、1,400 和 1,400 条。划分单位必须是 `canonical_pair_id`，并按任务类型、题型和数据来源保持分布基本一致。同一图像对、事件、近重复样本及其全部 TMPS 变式不得跨越三个训练子集。

```yaml
Round-1-Seed-SFT: 60%   # 4,200
Round-2-Rollout: 20%  # 1,400
Round-3-Rollout: 20%  # 1,400
```

#### 第一轮：初始 Tool-trace SFT

使用 Seed 60% 训练数据中经过确认的 canonical `gold_tool_trace` 对基础模型进行监督微调，得到初始 agent `Agent_0`。监督内容包括工具名称、调用参数、工具 observation 或中间结果引用，以及最终结构化答案。未经人工确认的 `silver_trace` 不应默认与 `gold_trace` 等权混合；如要使用，应单独标记、设置权重并进行消融。

该轮的目标是让 agent 获得基本的任务理解、工具选择、参数生成、工具调用格式和结果整合能力，为后续自主 rollout 提供稳定的初始策略。

#### 第二轮：第一次 Rollout 与定向更新

使用 `Agent_0` 在 Round-2 的 20% 新训练图像对及其适用的 TMPS 变式任务上进行 rollout。该子集的基础图像对及其所有变式在第一轮均不可见。每条 rollout 经工具真实执行、单任务 verifier 和 Cross-execution Verifier 检查后，划分为三类：

1. **正确且标准**：最终答案符合 schema，证据支持答案，轨迹可执行且关键依赖与参考路径一致，同时满足 TMPS 关系。该轨迹用于能力统计，通常不作为主要新增梯度数据。
2. **正确但路径不同**：最终答案正确，轨迹可执行且证据充分，但工具选择、调用顺序或参数策略与参考 trace 不同，并且满足全部硬约束和适用的 TMPS 关系。将其视为合法替代解法，构造成 `D_alt^(2)` 新增 SFT 数据。这里的“不同”指 trace 不同，而不是允许不符合 `output_schema` 的不同答案。
3. **错误或关系违反**：最终答案、工具调用、参数、中间证据、artifact 依赖或跨执行关系存在错误。定位首个可观测的关系违反节点，保留此前已经验证的正确前缀，在同一状态下产生并实际执行多个候选修复动作，构造 `D_pref^(2)` 局部偏好对。

第二轮的错误轨迹采用逐错修复，而不是在发现第一个错误后立即终止。修复当前节点后，从修复分支返回的真实 observation 和 artifact 状态继续 rollout；如果后续再次出现关系违反，则在新的共同前缀下继续构造偏好对，直到任务成功结束、达到调用预算或被标记为 `unresolved`。因此，一条 rollout 可以产生零个、一个或多个局部偏好对。

第二轮 SFT 数据必须同时包含新增合法路径和第一轮 Seed-SFT replay：

```text
D_sft^(2) = Weighted(D_alt^(2)) ∪ Replay(D_seed_sft, r_seed^(2))
```

`Replay(D_seed_sft, r_seed^(2))` 应从第一轮高质量 SFT 数据中按任务类型、输出 schema、关键工具和轨迹长度分层抽样，不能只随机重复高频模板。其作用是防止 agent 在集中学习新路径时遗忘 JSON/工具调用格式、基础工具参数、artifact 引用规则和标准工作流。`r_seed^(2)` 通过 Val 选择；初始实验可将 Seed replay 设为第二轮 SFT mini-batch 的 `10%–30%`，并报告实际比例和采样方法。

第二轮更新顺序为：

```text
Agent_0
→ Alternative-path SFT + Seed-SFT Replay
→ Local Counterfactual DPO on D_pref^(2)
→ Agent_1
```

DPO 阶段只使用共享同一正确前缀和环境状态、且候选分支均经过真实执行的 `chosen > rejected`。必要时保留一部分上一阶段偏好数据或格式样本进行 replay，但不能用未经执行的 LLM 判断直接生成偏好标签。

#### 第三轮：第二次 Rollout 与定向更新

使用更新后的 `Agent_1` 在 Round-3 最后一份 20% 新训练图像对及其适用的 TMPS 变式上重复第二轮流程。Round-3 的基础图像对及其所有变式在第一、二轮训练中均不可见，用于观察 Agent 经第一轮定向更新后在新场景中的剩余错误和新出现的能力缺口。

- 正确且标准的轨迹用于统计，通常不作为主要新增梯度数据；
- 正确但路径不同、证据充分且满足 TMPS 关系的轨迹进入 `D_alt^(3)`；
- 错误或违反关系的轨迹按照“定位当前首错 → 实际执行候选修复 → 从修复状态继续 rollout”的方式，形成 `D_pref^(3)`；
- 无法可靠定位、候选修复均失败或 verifier 相互冲突的样本标记为 `unresolved`，进入人工审计而不是强行构造偏好对。

第三轮 SFT 同样必须混入第一轮 Seed-SFT replay：

```text
D_sft^(3) = Weighted(D_alt^(3)) ∪ Replay(D_seed_sft, r_seed^(3))
```

必要时可再加入少量经过确认的第二轮合法替代轨迹，但第一轮 Seed-SFT replay 不能完全被第二轮数据替代，因为它承担基础格式、通用工具调用和标准工作流的稳定锚点作用。`r_seed^(3)` 仍通过 Val 选择，可从 `10%–30%` 起步搜索，并与第二轮分别报告。

第三轮更新顺序为：

```text
Agent_1
→ Alternative-path SFT + Seed-SFT Replay
→ Local Counterfactual DPO on D_pref^(3)
→ Agent_2
```

最终使用 Val 选择 `Agent_2` checkpoint、loss weight、replay 比例和 verifier 阈值。Test 只在所有方法和超参数冻结后进行最终评测；Test 上发现的失败不能继续生成偏好对并回流训练。

#### 首错节点与局部偏好的构造

对于错误 rollout，不将整条失败轨迹直接作为负样本，也不从头替换为唯一标准轨迹。设当前第一个可观测的关系违反节点为 `t`，训练样本保留 `t` 之前已经验证的正确轨迹前缀、真实 observation 和 artifact 引用，并在完全相同的状态下产生多个候选动作：

```text
正确前缀 + 当前 observation
├── chosen：能够修复当前错误并继续完成任务的工具调用
├── rejected-1：原始错误工具调用
├── rejected-2：工具正确但参数错误的调用
└── rejected-3：可执行但证据质量较差或成本过高的调用
```

只有经过实际工具执行和 verifier 验证的候选分支才能进入偏好数据。`chosen` 不只需要修复当前节点，还应在继续执行后获得更高的任务完成和关系满足分数；否则局部看似合理的动作可能把错误推迟到后续步骤。DPO 学习的是同一状态下的局部决策偏好，而不是对两条起点、observation 或 artifact 状态不同的轨迹进行无效比较。

修复当前节点后继续 rollout。如果在后续节点 `t+k` 出现新错误，则以修复后的前缀为新状态构造下一组偏好对。所有局部偏好样本都应保存来源 rollout、共同前缀、工具版本、输入/输出 artifact ID、候选动作的真实执行结果以及 verifier 判定，以便复现和审计。

#### 新增数据的加权策略

第二、三轮产生的新增 SFT 数据和 DPO 偏好对直接反映当前 agent 的能力缺口，因此可赋予高于 Seed 数据的单样本权重；但每轮 SFT 又必须保留足够的 Seed-SFT replay 作为稳定锚点。实现上可以组合 loss weighting、分层 sampler 或受控 oversampling：

```text
w_new_sft > w_seed_sft
w_new_dpo > w_replay
r_seed^(2) > 0
r_seed^(3) > 0
```

具体权重通过 Val 选择；新增样本权重可从 Seed 样本的 `2–5×` 开始搜索，Seed replay 比例可从每轮 SFT mini-batch 的 `10%–30%` 开始搜索。权重和比例不能混为一谈：即使 `w_seed_sft` 较低，也要保证 Seed 样本持续出现；即使 replay 比例较高，也不能让高频简单模板淹没新增的能力缺口。论文需要分别报告 `w_new_sft`、`w_seed_sft`、`r_seed^(2)`、`r_seed^(3)`、DPO/replay 采样比例和每类样本的实际更新次数。

推荐将主训练方法概括为：

> **Seed Trace-SFT + Seed-Replayed Alternative-path SFT + Relation-localized Counterfactual DPO**

该策略不依赖 PPO / GRPO 等在线强化学习，而是利用新样本上的真实 rollout、可执行 verifier、合法替代路径、关系违反节点反事实分支和 Seed-SFT replay，对 agent 的新增能力与实际错误进行定向更新，同时保持基本格式和工具使用能力。

7B 模型与 10k 数据规模总体匹配，但需要控制几个风险：

- 如果 trace 过于模板化，模型可能只记住固定工具顺序，而不是学习任务条件下的工具选择。
- 如果任务种类过多、schema 差异过大，7,000 条训练样本可能不足以覆盖全部组合。
- 如果 `gold_tool_trace` 噪声较高，7B 模型会较快学习错误模式，因此 trace 质量比数量更关键。
- Test set 应包含跨任务、跨传感器、错配准、云遮挡、缺失时相和长序列等情况，避免模型只在近似模板任务上表现良好。

## 遥感与时序专属机制

- **Temporal Evidence Planner**：先预测需要比较哪些时间步、哪些区域，再调用工具，避免盲目处理所有影像。
- **Geo-temporal Memory**：将每一步中间结果绑定到 `(time, geometry, sensor, CRS)`，防止 agent 混淆图像编号、坐标系和时间顺序。
- **Registration-aware Verification**：在变化检测前强制检查配准质量。若配准置信度低，则触发重配准或输出不确定性。
- **Change Hypothesis Loop**：agent 先生成变化假设，再通过检测、分割和统计工具验证，而不是一次性直接输出答案。
- **Evidence-to-Answer Constraint**：最终输出必须由中间工具结果支撑。若证据不足，在 schema 支持的情况下允许输出结构化的“无法可靠判断”。
- **Sensor-aware Tool Routing**：根据 RGB、Sentinel-2、多光谱、SAR 等元数据选择不同预处理、指数计算和分析工具。

## 测试计划

最终答案指标：

- 分类任务：accuracy / F1。
- 检测与变化区域：IoU、mask F1、polygon F1。
- 数值统计：relative error。
- 结构化问答：exact match，必要时加入 semantic judge。

Trace 指标：

- Tool-call precision / recall / order accuracy。
- Critical-step success rate，例如是否做配准检查、是否调用变化检测、是否进行证据验证。
- Executability：工具调用能否实际运行。
- Cost：调用次数、运行时间和 token 成本。

Evidence 指标：

- Key timestep accuracy。
- Evidence region IoU / F1。
- Evidence-answer consistency。

鲁棒性测试：

- 时间顺序打乱。
- 云遮挡或缺失时相。
- 轻微错配准。
- 跨传感器输入。
- 长序列中只有少数关键变化帧。

Ablations：

- 无 tool-trace SFT。
- 无 process preference。
- 无 verifier。
- 无 geo-temporal memory。
- 无 registration check。
- 只训练 answer，不训练 trace。

## 基本假设

- 目标是写成一篇 ML / CV / 遥感方向论文，而不是只做工程系统。
- 第一版 benchmark 不追求覆盖整个遥感领域，而是优先把“时序变化任务 + 可执行证据链”这条线讲清楚。
- 主要创新不是重新发明 VLM 架构，而是 benchmark、tool-trace supervision、agent post-training 和遥感时序验证机制。
- 工具可以先使用现成模型和传统遥感算子组合，论文亮点放在 agent 如何学会正确组织和验证这些工具。

## 数据样例

下面给出一个修订后的 xBD 灾害损毁评估样例。该样例遵循以下原则：

- 不设置 `difficulty` 字段；
- `temporal_inputs` 只包含 agent 可见信息；
- 特定任务可见的空间范围放入 `temporal_inputs`；
- `output_schema` 对 agent 可见；
- `gold_answer` 与 `output_schema` 格式严格一致；
- 不设置顶层 `gold_evidence` 字段；
- 不设置 `scoring_script` 字段。

```json
{
  "sample_id": "xBD_hurricane_000137",
  "task_type": "disaster_damage_assessment",
  "source_dataset": "xBD",
  "task_instruction": "Compare the pre-disaster and post-disaster images within the given region of interest. Count the buildings that are damaged after the event and report the number of buildings by damage level.",
  "temporal_inputs": {
    "region_of_interest": {
      "type": "Polygon",
      "coordinates": [
        [
          [-95.3731, 29.7608],
          [-95.3714, 29.7608],
          [-95.3714, 29.7594],
          [-95.3731, 29.7594],
          [-95.3731, 29.7608]
        ]
      ]
    },
    "images": [
      {
        "image_id": "pre_event_image",
        "timestamp": "2017-08-25",
        "role": "pre_event",
        "uri": "images/xBD_hurricane_000137_pre.tif"
      },
      {
        "image_id": "post_event_image",
        "timestamp": "2017-08-31",
        "role": "post_event",
        "uri": "images/xBD_hurricane_000137_post.tif"
      }
    ],
    "event": {
      "event_type": "hurricane",
      "event_name": "Hurricane Harvey"
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "total_building_count": {
        "type": "integer"
      },
      "damage_counts": {
        "type": "object",
        "properties": {
          "no_damage": {
            "type": "integer"
          },
          "minor_damage": {
            "type": "integer"
          },
          "major_damage": {
            "type": "integer"
          },
          "destroyed": {
            "type": "integer"
          }
        },
        "required": ["no_damage", "minor_damage", "major_damage", "destroyed"]
      },
      "damaged_building_count": {
        "type": "integer"
      }
    },
    "required": ["total_building_count", "damage_counts", "damaged_building_count"]
  },
  "gold_answer": {
    "total_building_count": 18,
    "damage_counts": {
      "no_damage": 13,
      "minor_damage": 2,
      "major_damage": 3,
      "destroyed": 0
    },
    "damaged_building_count": 5
  },
  "gold_tool_trace": [
    {
      "step_id": "s1",
      "tool": "load_temporal_images",
      "arguments": {
        "image_ids": ["pre_event_image", "post_event_image"],
        "region_of_interest": "temporal_inputs.region_of_interest"
      },
      "outputs": {
        "pre_crop_id": "crop_pre_000137",
        "post_crop_id": "crop_post_000137"
      }
    },
    {
      "step_id": "s2",
      "tool": "inspect_raster_metadata",
      "arguments": {
        "raster_ids": ["crop_pre_000137", "crop_post_000137"]
      },
      "outputs": {
        "metadata_report_id": "metadata_000137"
      }
    },
    {
      "step_id": "s3",
      "tool": "check_coregistration",
      "arguments": {
        "reference_raster_id": "crop_pre_000137",
        "target_raster_id": "crop_post_000137"
      },
      "outputs": {
        "registration_report_id": "registration_000137",
        "registration_status": "pass"
      }
    },
    {
      "step_id": "s4",
      "tool": "extract_buildings",
      "arguments": {
        "raster_id": "crop_pre_000137"
      },
      "outputs": {
        "building_layer_id": "buildings_pre_000137"
      }
    },
    {
      "step_id": "s5",
      "tool": "classify_building_damage",
      "arguments": {
        "pre_raster_id": "crop_pre_000137",
        "post_raster_id": "crop_post_000137",
        "building_layer_id": "buildings_pre_000137"
      },
      "outputs": {
        "damage_layer_id": "damage_000137"
      }
    },
    {
      "step_id": "s6",
      "tool": "summarize_damage_counts",
      "arguments": {
        "damage_layer_id": "damage_000137",
        "region_of_interest": "temporal_inputs.region_of_interest"
      },
      "outputs": {
        "total_building_count": 18,
        "damage_counts": {
          "no_damage": 13,
          "minor_damage": 2,
          "major_damage": 3,
          "destroyed": 0
        },
        "damaged_building_count": 5
      }
    },
    {
      "step_id": "s7",
      "tool": "validate_output_schema",
      "arguments": {
        "candidate_answer": "s6.outputs",
        "output_schema": "output_schema"
      },
      "outputs": {
        "valid": true
      }
    }
  ]
}
```

## 样例备注

1. `temporal_inputs` 是灵活字段。对于 xBD 类数据，它可以包含时间戳、图像角色、事件元数据、图像 URI 和 ROI。对于 Sentinel / Landsat 类任务，它也可以包含波段列表、产品标识符、采集日期或云掩膜资产标识符，前提是这些信息在任务设定中对 agent 可见。
2. CRS、分辨率、geotransform、nodata value 和配准质量等字段没有放入该样例的 `temporal_inputs`，因为这些信息应由 agent 通过 `inspect_raster_metadata`、`check_coregistration` 等工具探索得到。
3. `output_schema` 是模型可见字段，用于定义最终输出格式。它只保留评估所需字段，避免加入 `summary`、`confidence`、`key_timestep` 等自然语言或弱相关字段。
4. `gold_answer` 必须严格遵循 `output_schema`，这样可以降低答案评估的歧义，并便于程序化评分。
5. `gold_tool_trace` 中可以包含 `registration_status`、`damage_layer_id` 等中间输出。这些不是模型输入，而是用于 trace 监督或过程评估的目标。
6. benchmark 不应只使用 exact trace match 作为过程评价标准。时序遥感任务可能存在多条合理工具路径，因此之后的 trace 评价应考虑关键步骤覆盖率、可执行性、工具参数合法性、证据-答案一致性和成本。
7. 不设置顶层 `gold_evidence` 字段。证据可以通过可执行工具输出隐式表达，例如 mask、polygon、渲染图层或统计结果，并由 trace 步骤引用。
