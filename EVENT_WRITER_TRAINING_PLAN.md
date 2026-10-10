# 事件语料 writer 第一次短训练：100步固定试跑

## 执行前冻结

用户已授权下一步训练。使用hku-gpu2双5090：GPU0训练、GPU1评估；原始Qwen3-4B，不使用退化的1B骨干初始化。此轮只做阶段A的单块warmup，不训练遗忘控制器。

- 数据来自event_curriculum_v1的64条train/16条dev warmup，共8/2世界组、16/4唯一context；test不读取分词、不模型评分。组内4问题×2twin相关，不能当独立世界。
- 每条材料装成恰好4096 token。真实事实及首块记录作为完整后缀，不裁剪；前面使用按世界生成、twin共用的唯一合成维修记录背景。不是FineWeb/LongCrawl；自然背景接入延后到学习通路验证之后。
- query加固定`\nAnswer:`，拼接` 空格+答案`联合分词，严格验证prompt token前缀不跨答案边界；答案多token全部监督，另加一个终止EOS。prompt/background不算答案loss，未来文本和family/事件真值不进入前向。
- 原始36层骨干全部冻结；7层[0,6,12,18,24,30,35]原生conv/proj共14张量为FP32 master，其余BF16；BF16 autocast、SDPA、关闭TF32。训练目标是conv/proj本身，不是对detach候选的LowRankWriter变换。
- 完整一块通过原有cbf_forward_mlp产生原生公式delta，不调用inference_mode或detach；丢弃写入KV，以fresh-KV读取delta并反传答案loss。单块输出不依赖同块尚未提交的候选，故写入特征仍是冻结基座的上下文特征。当前只验证单块，不宣称实现多次决策的端到端展开。
- 初始化匹配baseline：conv=0，proj~N(0,initializer_range)，seed301，ttt_lr=.3。第一步proj梯度为0属于乘积结构预期，conv必须非零；之后两类梯度均要求有限非零。
- AdamW lr=1e-7、weight_decay=0、clip=1、固定100步；每步一条query，64训练条目按seed洗牌循环，不看dev调整lr/步数。FP32 optimizer state，最终保存writer与optimizer，可复核冻结骨干hash。
- 数值失败/OOM/梯度断开即保留失败工件并停止；delta相对原down_proj范数>1是预设停止线，无额外裁剪delta或事后改尺度。无自动超参搜索。

## 固定评估

保存0/25/50/100步writer。GPU1读取完整ready/hash后评估全部16 dev问题：

1. 正确context生成的快记忆，fresh KV。
2. 空记忆，fresh KV。
3. 同一split不同世界的快记忆，固定按组排序循环错配，fresh KV。
4. 完整context KV、不写入快记忆，普通阅读参考。
5. 另对4个anchor问题评估单事实twin记忆，其他问题不把答案未改变的twin当错误记忆。

主指标为真实答案多token全词表NLL（不含终止EOS），以及最多16个新token、遇EOS停止、去首尾空白后完整字符串EM。记录首token命中、首token `<|im_end|>`概率，禁止屏蔽结束token或只在候选内排名。teacher forcing损失包含一个答案后EOS，另报EOS NLL；自由生成无答案前缀泄露。全部逐查询/写入/训练步记录耗时和allocated/reserved显存峰值。

固定终点100步判断，不选表现最好的中间点：对空记忆和错配记忆均有NLL改善≥0.05，EM提升≥12.5百分点（16题中2题），两个dev世界组各有正NLL增益，才标记初步有用记忆信号。此门槛是小规模机制判据，不是显著性或泛化证明。不通过仍完整报告100步结果，不延长到通过。

末尾另对按组ID预先排序的前两个train世界的16条问题测正确记忆，区分训练学习与dev泛化；该probe未参与选择。初始零记忆三对照须逐聚合一致，所有checkpoint的empty/full-KV指标须一致，骨干hash须与此前原始模型审计相同。

## 代码、兼容性与复现

新增独立文件：`tasks/pack_cbf_event_warmup.py`、`cbf_ttt/event_writer.py`、`tasks/train_cbf_event_writer.py`、`tests/test_cbf_event_writer.py`、`scripts/run_cbf_event_writer.sh`。复用runtime原生candidate公式与已有FP32 writer选择，不修改baseline代码/配置，无新依赖。

```bash
cd /home/ctj/cbf_ttt_joint_exp_20260927
ROOT=/home/ctj/cbf_ttt_event_writer_100step_v1 bash scripts/run_cbf_event_writer.sh
```

ROOT/SOURCE/PYTHON/MODEL控制路径；矩阵和100步预算冻结，修改研究设置需新目录/记录。默认SOURCE=/home/ctj/cbf_ttt_event_curriculum_v1，MODEL=/home/ctj/models/Qwen3-4B。拒绝覆盖已有输出。

本地9项测试中6通过、3无torch跳过；服务器运行全部9项后才启动。测试覆盖真实非零writer梯度、零初始化的预期、原生单块delta和fresh-KV读取对齐、答案位置/EOS、pack完整事实/未来test不读取。GPU结果、资源与结论待实际完成。

风险：只有8训练世界与2验证世界；code字符串共享格式可能产生语言格式学习而非内容记忆，所以必须看错配、twin及完整EM；合成背景可能稀释事实；100步可能不足；4096块与原始短事实分布差异大。结论必须区分训练loss下降、格式学习、内容记忆和泛化。只有本轮有用记忆信号成立，后续才进入多块retain、遗忘动作标签和联合训练。
