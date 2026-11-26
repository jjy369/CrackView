# 路面裂缝三维重建视图筛选 · R3
从图像、相机外参和已有的重建质量标签出发，训练评分网络与 PPO，并为每个待测场景输出 25 对图像和相机文件。本文档中的命令均从解压后的 R3 根目录执行。需要 Python 3.10+，先安装与显卡兼容的 torch、torchvision，再运行 `pip install -r requirements.txt`。

## 代码结构
| 路径 | 用途 |
|---|---|
| score/build_manifest.py、dataset.py | 对齐图像和外参，按场景划分训练与验证集，生成 13 维位姿并做视图丢弃增强 |
| score/model.py、train_scoring_net.py | ResNet18、视图 Transformer、8 维覆盖描述、场景 Transformer 和双输出质量回归 |
| feature/train_dino_vit_crack.py | DINO 自监督训练及基于 YOLO 框的裂缝辅助训练 |
| feature/make_sfm_features.py | 从完整视图库生成 397 维 NPZ，部署时无需标注 |
| SfM_PPO_env.py | 场景观测、12 头 MMR、候选联合评分和逐候选奖励 |
| SfM_PPO_interaction.py、SfM_train.py | 单步 PPO、断点续训、随机十候选择和结果导出 |
| benchmark/quality.py | 输入配准后的点云与 ICP 指标，计算式（4）至（10）的监督分数 |
| smoke_test.py | 两个合成场景上的评分反向传播、PPO 更新与推理检查 |

## 输入
评分数据沿用原工程布局：每个“采样策略/场景”目录含 images 和 cams。图像 0001.jpg 对应 0001.txt 或 0001_cam.txt；外参使用含 extrinsic 标记的世界到相机矩阵。质量表至少含 status、pred_folder_key、global_q_geo_raw、crack_core_q_geo_raw、q_final。R3 实验的注册可靠性均为 1。已有注册失败样本应以有效零标签记录并保留对应图像和外参。

特征清单使用 scene_name、image_paths、camera_paths，后两列分别以分号连接对应路径；可用同场景各子集的并集，也可每个场景直接提供一行完整的 50 对路径。每个场景的图像文件主名应唯一。请将原始拍摄序列与独立测试场景分开，评分模型、DINO 和 PPO 的训练均不使用测试场景。220 个开发场景按 176/44 划分，另外 22 个测试场景需从外部提供。

## 运行
以下路径是占位示例。评分模型首次使用 ImageNet 预训练权重，需要可用的本地缓存或下载条件。

~~~shell
python -m score.build_manifest --root subset_data --scores point_score.csv
python -m score.train_scoring_net --checkpoint_root checkpoints_r3 --seed 2024
python -m feature.train_dino_vit_crack --images train_images --output weights/dino
python -m feature.train_dino_vit_crack --images train_images --labels train_labels --backbone weights/dino/backbone.pth --output weights/crack
python -m feature.make_sfm_features --manifest data/train.csv --backbone weights/crack/backbone.pth --out features/train.npz
python -m feature.make_sfm_features --manifest data/test_pool.csv --backbone weights/crack/backbone.pth --out features/test.npz
~~~

在 checkpoints_r3/seed2024 内找到 best_top1_ 开头的 .pth，将下例 SCORER.pth 换成该文件的完整路径。正式训练使用 800 个 rollout，每个 rollout 采样 32 个主场景上下文，每个上下文保留 10 条动作样本，共 320 条。推理输出每个场景的 10 组三项分数、胜出候选编号以及对应的 25 对文件路径。评分数组依次为全局、裂缝核心和派生综合分。

~~~shell
python SfM_train.py train --features features/train.npz --scorer SCORER.pth --policy weights/policy.pt
python SfM_train.py infer --features features/test.npz --scorer SCORER.pth --policy weights/policy.pt --out selected_views.json
python -m smoke_test
~~~

训练中断后在同一训练命令末尾加 --resume。旧版三输出评分权重和旧 PPO 权重的形状与本工程不同，需要重新训练。历史 397 维特征只有在骨干与几何、质量编码均一致时才适合复用。

## 公式及实现约定
- 式（11）使用旋转矩阵的九个元素、组内归一化的相机中心三个分量和一个半径。几何统计排除 padding。
- 式（16）至（19）只对两个分量执行 Smooth-L1，权重分别为 0.4 和 0.6。排序与 Top-1 都用派生综合分；预测头共有两个输出。
- 式（28）至（35）在主、辅场景分别对同一批十个动作执行 MMR。各场景的十个子集一次进入场景 Transformer，候选间共享上下文。训练使用每个候选的回报减去主场景 critic 估值；推理只取综合分最高的候选。
- R3 未指定四个随机基线子集的陪衬候选。本实现让四个随机子集在独立的上下文中联合评分，前八次访问取运行均值，随后以 0.1 更新。十个候选共享同一次场景统计更新，以免候选排列影响回报。
- 验证和测试特征使用 PPO 训练时保存的标准化参数。新场景的历史质量与难度从零开始。默认推理种子为 2024，可用 --seed 更换。
- benchmark/quality.py 使用米为单位的 N×3 NPY 坐标。reference 为全视图参考点云，core 为从参考点云中标注的裂缝核心，pred 为已对齐的子集点云；fitness 与 inlier-rmse 由配准流程提供。实测论文指标还需原始点云、人工标注、Metashape 工程和独立测试清单。包中不含这些数据，也不含论文实验权重。

## Metashape 重建
在装有 Metashape Professional 2.x 及其 Python 模块的环境中运行 python benchmark/reconstruct.py --selected selected_views.json --out clouds。全视图参考亦可使用相同 JSON 格式并列入所有 50 个视图。该适配脚本尚未在持有许可的 Metashape 环境中执行；导出坐标应先与米制参考配准，再计算质量分。
