# RSRCC 专用工具注册表

## 1. 适用边界

本注册表仅面向 RSRCC 的双时相 RGB 图像问答任务。当前本地测试集 `RSRCC/test/metadata.csv` 含 21,999 道题：Yes/No 与 A–D 选择题并存。图像为 512×512 三通道 RGB，因此本表不将多光谱指数、云掩膜、CRS 或真实地理面积作为 RSRCC 的必需能力。

原始 `text` 同时包含 Question 和 Answer。构造 agent 输入时必须先拆出 `question`、候选选项和 `answer_format`；`Answer` 只能保留在 `gold_answer`，不得传给 agent。

## 2. 场景 ontology

| 对象组 | 必须覆盖的对象或别名 | 典型变化 |
|---|---|---|
| 建筑与构筑物 | building、house、structure、shed、warehouse、commercial building、apartment、mobile home、container、greenhouse、gazebo、roof、foundation、debris | 新建、拆除、毁坏、扩建、替换、屋顶颜色或施工状态变化 |
| 道路与线状设施 | road、street、sidewalk、driveway、trail、path、bike lane、railway、bridge、intersection、cul-de-sac、roundabout、crosswalk | 新建、移除、延伸、变宽、变窄、铺装、阻塞 |
| 停车与硬化地表 | parking lot、parking stalls、pavement、paved pad、impervious surface | 新建、扩张、标线变化、车辆数量或布局变化 |
| 植被 | tree、tree cluster、vegetation、forest、woodland、grass、lawn、shrub、garden、landscaped island | 生长、种植、清除、砍伐、变密、被其他对象取代 |
| 水体与休闲设施 | water、pond、lake、stream、canal、swimming pool、tennis court、athletic field、stadium、playground | 出现、消失、蓄水、退水、覆盖、转换 |
| 小型附属对象 | vehicle、car、truck、boat、fence、wall、solar panel、construction equipment | 出现、移除、数量或排列变化 |
| 地表与地块 | field、agricultural land、open land、vacant lot、dirt、bare ground、terrain | 开发、铺装、绿化、变成建筑/道路/停车场/水体 |
| 全局场景 | scene consistency、global alteration、anomaly、visual variance、temporal deviation | 无变化、局部变化、全局变化 |

## 3. 工具注册表

### 实现类型

- **直接调用**：可直接调用成熟开源库；项目只需封装稳定的输入输出。
- **框架 + 模型**：调用现有框架，但必须固定或训练适配 RSRCC ontology 的权重；框架本身不能保证任务可用。
- **项目实现**：需要在项目中实现规则、artifact 管理或多个库/模型结果的编排；不是从零重写底层视觉算法。

| 工具 | 调用条件 | 功能 | 输入 | 输出 artifact | 实现类型与建议后端 |
|---|---|---|---|---|---|
| `load_rgb_pair` | 必须 | 加载 before/after，建立不可变样本 artifact | `img_1`, `img_2` | `RGBPair` | **直接调用**：Pillow 或 OpenCV；项目封装 artifact ID |
| `inspect_rgb_pair` | 必须 | 检查尺寸、通道、编码、像素范围、损坏文件 | `RGBPair` | `PairMetadataReport` | **直接调用**：Pillow/OpenCV；项目定义检查阈值 |
| `check_pixel_alignment` | 必须 | 测量两期图像的平移、缩放与配准可信度 | `RGBPair` | `AlignmentReport` | **直接调用**：OpenCV ECC/feature matching；项目定义 pass/fail 规则 |
| `correct_pixel_alignment` | 仅当检查失败 | 校正轻度像素错位 | `RGBPair`, `AlignmentReport` | `AlignedRGBPair` | **直接调用**：OpenCV warp；项目限制变换类型并保存矩阵 |
| `detect_multiscale_change` | 必须 | 输出全图和局部的变化 mask、置信度、连通区域 | `AlignedRGBPair` | `ChangeMap`, `ChangeRegions` | **框架 + 模型**：Open-CD；需在 RSRCC/LEVIR 类 RGB 对上选定 checkpoint、阈值和版本 |
| `parse_scene_entities` | 必须 | 解析建筑、植被、水体、地表、设施等语义与实例 | 单期 RGB、ontology 版本 | `SceneLayer`, `EntityInstances` | **框架 + 模型**：TorchGeo/MMSegmentation/MMDetection 均可作为运行框架；必须训练或微调覆盖第 2 节所有类别的本地模型 |
| `extract_linear_infrastructure` | 道路、人行道、小径、铁路等题 | 精细提取线状设施、中心线与连通性 | RGB, `SceneLayer` | `LinearFeatureLayer`, `NetworkGraph` | **框架 + 模型**：分割模型 + OpenCV/GeoPandas skeletonization；人行道/斑马线需专门标注或微调，不能仅依赖 change detector |
| `parse_paved_and_parking_areas` | 停车场、铺装、车位题 | 区分道路、停车场、广场、硬化地和标线 | RGB, `SceneLayer` | `PavedAreaLayer`, `ParkingLayout` | **框架 + 模型**：语义分割模型 + 几何后处理；需含 parking/road/pavement 类训练数据 |
| `detect_small_aux_objects` | 车辆、船、集装箱、围栏等题 | 检测小目标并计数/定位 | RGB, ROI | `SmallObjectLayer`, `ObjectCounts` | **框架 + 模型**：MMDetection/YOLO 类检测器；需以 RSRCC 分辨率验证 vehicle、container、fence、solar 等类别 |
| `match_entities_across_time` | 必须 | 匹配两期对象，标记新增、消失、持续、分裂/合并 | before/after `EntityInstances` | `TemporalEntityMatches` | **项目实现**：IoU、距离、类别和图结构约束的匹配器；底层可用 NumPy/Shapely |
| `compare_entity_attributes` | 扩建、屋顶、树木、水体等题 | 比较面积、形状、RGB 色彩、纹理、树冠/水面覆盖 | 匹配对象、RGB 对 | `AttributeDeltaTable` | **项目实现**：OpenCV/NumPy 特征计算 + 明确规则；不依赖通用 VLM 直接作答 |
| `build_scene_graph` | 必须 | 构建对象及 `north_of`、`near`、`inside`、`adjacent_to` 等关系 | scene/linear/small-object layers | `SceneGraph` | **项目实现**：Shapely/GeoPandas/NetworkX；关系阈值、方位和序数规则固定版本 |
| `resolve_spatial_query` | 必须 | 用结构化空间条件找到题目所指目标或 ROI | `QuerySpec`, `SceneGraph` | `QueryROI`, target IDs | **项目实现**：约束求解器；agent 负责把问题解析为 `QuerySpec`，工具负责可复现的几何定位 |
| `infer_change_event` | 必须 | 由变化、匹配、属性证据生成规范事件 | `QueryROI` 和上游 artifacts | `EventEvidence` | **项目实现**：事件规则引擎，输出受控事件词表，禁止直接凭自然语言猜测 |
| `summarize_roi_change` | 必须 | 输出像素面积、比例、长度、对象数与前后差值 | ROI、事件相关 artifacts | `ChangeStatistics` | **直接调用 + 封装**：NumPy/OpenCV/GeoPandas；只报告 pixel/ratio/count，不报告平方米 |
| `score_answer_options` | A–D 题 | 将每个候选选项转为事件约束并按证据评分 | options, `EventEvidence` | `OptionScores` | **项目实现**：选项解析模板 + 事件/属性约束比对；保留每项支持与冲突证据 |
| `assess_global_scene_change` | 全局一致性题 | 判定无变化、局部变化或全局变化 | `ChangeMap`, statistics | `GlobalChangeReport` | **项目实现**：基于变化面积、区域数和分布的阈值规则，阈值由开发集固定 |
| `render_evidence` | 必须 | 输出 before/after、ROI、对象、变化层、锚点与统计图 | 相关 artifacts | `EvidenceRender` | **直接调用 + 封装**：Matplotlib/OpenCV；项目定义统一图例和文件命名 |
| `verify_evidence_consistency` | 必须 | 检查答案、空间位置、对象、事件和统计是否相互支持 | candidate answer, evidence artifacts | `VerificationReport` | **项目实现**：规则 verifier；Yes/No 和 A–D 走不同验证分支 |
| `validate_output_schema` | 必须 | 验证最终输出只能是 `yes/no` 或一个 `A/B/C/D` | candidate answer, `output_schema` | `SchemaValidationReport` | **直接调用**：Pydantic/JSON Schema |

## 4. 受控事件词表

`infer_change_event` 的输出仅允许使用下列事件：

| 事件族 | 事件 |
|---|---|
| 新增/消失 | `constructed`, `appeared`, `installed`, `removed`, `demolished`, `destroyed`, `cleared` |
| 形状变化 | `expanded`, `enlarged`, `extended`, `widened`, `narrowed` |
| 替换与转换 | `replaced`, `converted`, `paved`, `unpaved`, `developed` |
| 自然变化 | `grown`, `planted`, `vegetation_increase`, `vegetation_loss`, `water_increase`, `water_decrease` |
| 外观变化 | `repainted`, `roof_color_changed`, `surface_covered`, `construction_completed` |
| 小目标变化 | `object_added`, `object_removed`, `count_increased`, `count_decreased`, `layout_changed`, `obstructed` |
| 场景状态 | `unchanged`, `local_change`, `global_change` |

## 5. 执行闭环与不可省略约束

```text
question + RGBPair
  -> load_rgb_pair / inspect_rgb_pair / check_pixel_alignment
  -> detect_multiscale_change + parse_scene_entities
  -> required specialized parsers
  -> build_scene_graph -> resolve_spatial_query
  -> match_entities_across_time / compare_entity_attributes
  -> infer_change_event -> summarize_roi_change
  -> score_answer_options or Yes/No decision
  -> render_evidence -> verify_evidence_consistency -> validate_output_schema
```

1. 所有工具都引用 artifact ID，不能把大图像或隐藏答案直接写回 observation。
2. `parse_scene_entities`、`extract_linear_infrastructure`、`parse_paved_and_parking_areas` 和 `detect_small_aux_objects` 必须各自通过 RSRCC 开发集验收；否则不得标记为可用后端。
3. `resolve_spatial_query` 是 RSRCC 必需工具：题目大量使用方向、相邻、参照物和序数描述；只做全图变化检测不能可靠作答。
4. 全局一致性题可省略对象匹配，但仍须执行变化检测、全图统计和 verifier。
5. 工具轨迹按必需步骤与偏序评估；合理的不同执行顺序不应被当作错误。

## 6. 调用来源与版本锁定

- 图像 I/O、像素级配准、属性计算、渲染：Pillow、OpenCV、NumPy、Matplotlib。
- 变化检测：Open-CD，作为模型运行框架；固定模型配置、checkpoint 哈希、阈值与推理设备。[Open-CD](https://github.com/likyoo/open-cd)
- 多类别语义/实例模型：TorchGeo、MMSegmentation、MMDetection 可作为框架；它们不提供开箱即用的完整 RSRCC ontology，必须由项目维护本地模型适配器。[TorchGeo](https://torchgeo.org/)；[MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- 场景图、空间关系与统计：Shapely、GeoPandas、NetworkX；这些库只提供几何原语，RSRCC 的关系语义由项目实现。
- schema 验证：Pydantic 或 JSON Schema。

因此，项目不需要从零实现图像处理、配准、神经网络推理、几何运算或 JSON 校验；但必须自行实现统一工具接口、artifact 存储、RSRCC ontology 适配、时相对象匹配、空间查询、事件推断、选项评分和证据 verifier。最主要的研发工作是训练/微调覆盖 RSRCC 类别表的本地视觉模型，而不是重复造底层库。

## 7. 当前实现状态（2026-10-18）

实现位于 `src/teo_agent/rsrcc/`，测试位于 `tests/`。第 3 节列出的 20 个工具均已注册到受限执行器，并具有基础单元测试。执行器只向 controller 返回 artifact ID 和小型 JSON observation；RGB、mask 和渲染图等大对象保存在不可变、内容寻址的 `ArtifactStore` 中。

| 部分 | 当前状态 | 说明 |
|---|---|---|
| 图像加载、检查、配准检查/校正 | 已实现、基础测试通过 | 使用 Pillow/NumPy；执行有限的各向同性尺度搜索与平移估计/校正，不宣称处理复杂仿射或非刚性误差 |
| 变化检测 | 基础后端已实现、生产后端接口已定义 | 默认是 NumPy 多尺度差分，仅用于执行链调试；`CallableChangeBackend` 可封装锁定 checkpoint 的 Open-CD 推理器 |
| 实体、线状设施、停车区、小目标解析 | 基础后端已实现、生产后端接口已定义 | 默认 RGB 启发式不具备 RSRCC 验收资格；`CallableEntityBackend` 可封装 MMSegmentation/MMDetection 等本地模型 |
| 时相匹配、属性比较、场景图、空间查询 | 已实现、基础测试通过 | 输出匹配、增删、分裂/合并候选、属性差值、关系图和 QueryROI artifact |
| 事件推断、区域统计、选项评分、全局判断 | 已实现、基础测试通过 | 事件输出受第 4 节词表限制；统计仅使用 pixel、ratio、count、pixel length |
| 证据渲染、证据一致性和输出 schema | 已实现、基础测试通过 | 分别生成 PNG artifact、规则验证报告和严格的 `yes/no` 或 `A/B/C/D` 校验结果 |
| RSRCC 标注拆分 | 已实现、全量本地测试通过 | 本地 test split 的 21,999 条标注均可拆分；`agent_view()` 不包含 gold answer |

### 状态边界

“基础测试通过”只证明接口、artifact 流转和确定性规则在合成样例上可执行，不代表视觉精度已经满足 RSRCC。默认变化与实体后端均返回 `validated_for_rsrcc: false`。生产后端只有在记录 checkpoint SHA-256、预处理版本并通过独立 RSRCC 开发集验收后，才能设为 `validated_for_rsrcc: true`；代码会拒绝没有 checkpoint 哈希却声称已验收的后端。

运行全部测试：

```bash
PYTHONPATH=src python3 -m unittest discover -v
```

## 8. 随机 100 题实测（2026-10-18）

固定 `seed=20261018` 从本地 test split 无放回抽取 100 题，由纯文本控制器读取问题与结构化 observation 并真实调用本表工具。最终 100/100 执行成功、20/20 个注册工具均被调用、严格正确 75/100；但 20 个样本命中显式工具能力缺口，且全部样本仍依赖未经 RSRCC 验收的基础变化检测与实体后端。因此当前状态是“接口闭环覆盖”，不是“生产语义完全覆盖”。完整方法、trace、缺口和审计见 [`experiments/rsrcc_100_seed_20261018/report.md`](experiments/rsrcc_100_seed_20261018/report.md)。
