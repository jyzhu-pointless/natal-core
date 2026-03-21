# TODO

- `Species.from_dict` 构建后，应当支持指定性染色体
- 检查性染色体的遗传逻辑是否正确（遗传张量雌雄 index 应该不同）

你提的第二点我也核对了：

initialization / finish 现状

事件定义里仍有 initialization、finish（以及 first/early/late）。
base_population.py (line 51)
types.py (line 124)
kernel 加速路径目前只执行 first/early/late（CSR+chain）。
simulation_kernels.py (line 382)
finish 是 Python 层触发（run 结束或 finish_simulation()），不在 kernel 事件链里。
age_structured_population.py (line 878)
discrete_generation_population.py (line 233)
base_population.py (line 801)
initialization 目前也在 Python 事件体系里，不在 kernel 执行路径。
如果你后面要，我可以给你一版“不依赖全局激活状态、同时保留缓存友好”的并发安全 ABI 方案。



