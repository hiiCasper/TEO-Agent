# Temporal EO Agent：RSRCC 工具实现

本项目实现 `rsrcc-tool-registry.md` 定义的 20 个能力级工具，用于构造可执行、可验证的 RSRCC 双时相 RGB 工具轨迹。

## 安装与测试

```bash
python3 -m pip install -e .
python3 -m unittest discover -v
```

项目运行时只强制依赖 NumPy 和 Pillow。生产视觉模型通过后端适配器接入，因此核心 runner 不依赖特定的训练框架。

## 最小调用示例

```python
from teo_agent.rsrcc import RsrccTools, ToolExecutor, parse_rsrcc_annotation

tools = RsrccTools()
executor = ToolExecutor(tools, max_calls=32)

pair = executor.call(
    "load_rgb_pair",
    {"img_1": "before.png", "img_2": "after.png"},
).artifacts["rgb_pair_id"]

metadata = executor.call("inspect_rgb_pair", {"rgb_pair_id": pair})
alignment = executor.call("check_pixel_alignment", {"rgb_pair_id": pair})
changes = executor.call("detect_multiscale_change", {"rgb_pair_id": pair})
entities = executor.call(
    "parse_scene_entities",
    {"rgb_pair_id": pair, "time": "after"},
)
```

每次调用返回 `ToolResponse`：

- `artifacts`：后续工具可引用的 artifact ID；
- `observation`：可安全交给文本 controller 的小型结构化结果，不含完整影像或 mask。

`executor.trace_as_dicts()` 可直接生成带参数、artifact 引用、observation 和耗时的可执行 tool trace。

## 接入本地模型

默认视觉后端是可测试的确定性 baseline，不应作为 RSRCC 精度结果。Open-CD、MMSegmentation、MMDetection 或其他本地模型可以通过 `CallableChangeBackend` 和 `CallableEntityBackend` 接入：

```python
from teo_agent.rsrcc import CallableChangeBackend, RsrccTools

change_backend = CallableChangeBackend(
    predictor=local_model_predict,
    name="opencd-rsrcc-v1",
    checkpoint_sha256="<64-character checkpoint digest>",
    preprocessing_version="rsrcc-rgb-512-v1",
    validated_for_rsrcc=True,
)
tools = RsrccTools(change_backend=change_backend)
```

若 `validated_for_rsrcc=True` 却没有 checkpoint 哈希，工具会拒绝执行。开发集的指标和阈值仍需随真实 checkpoint 一起登记；基础单元测试不等于视觉模型验收。

## 代码结构

- `src/teo_agent/rsrcc/artifacts.py`：不可变、内容寻址 artifact；
- `src/teo_agent/rsrcc/backends.py`：变化/实体视觉后端协议与 baseline；
- `src/teo_agent/rsrcc/tools.py`：20 个 RSRCC 工具；
- `src/teo_agent/rsrcc/runner.py`：allowlist、参数检查、调用预算和 trace；
- `src/teo_agent/rsrcc/parsing.py`：RSRCC 标注拆分与答案防泄漏；
- `tests/`：逐工具测试、执行器测试及 21,999 条本地标注契约测试。
