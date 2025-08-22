# Meteor_detection
- 一个基于深度学习的流星轨迹识别,在风光摄影场景下,筛选出大量星空照片中含有流星轨迹的部分
# 使用方法
- 在./Camera_jpg_data下存放需要筛选的图像(jpg格式),直接运行./Detection.py即开始处理,结果输出于./result

- 由于Github无法上传较大文件,故模型参数已被拆分为多个小文件,位于./model_parts下,首次使用,需要运行./merge_model.py以合并为需要的参数文件,届时将出现./model.pth

# 运行逻辑
- ./Camera_jpg_data将首先通过图像预处理pipeline,包括图像增强与阈值分割(由./image_processing_pipeline.py定义),随后由模型筛选

# 模型训练
- 考虑到实际场景需求,在训练中强化了对正类样本的权重
- 为提高流星检测的召回率，采用低决策阈值：
- 设置 prediction_threshold=0.3 而非常规的0.5，优先保证正类样本的识别
- 在评估函数中使用配置的阈值替代硬编码值，提高灵活性
- 混合精度训练
- 梯度累积
- 学习率自适应调度
- 早停机制 ：early_stopping_patience=10 , early_stopping_delta=0.001
- 权重衰减正则化

# 模型架构

- Train_val__test_meteor_detection_cnn.py 文件中实现了名为 OptimizedMeteorCNN 的卷积神经网络模型，该模型具有以下核心特色：

核心架构特点：

- 5层递进式卷积结构，逐层加深特征提取能力（64→128→256→384→512通道）
- 采用递减的卷积核尺寸设计（11×11→7×7→5×5→3×3→3×3）
- 每层卷积后均配置批量归一化（Batch Normalization）加速收敛
- 所有ReLU激活函数使用 inplace=True 优化内存使用
- 包含自适应平均池化层( AdaptiveAvgPool2d )，确保输出尺寸一致性
- 全连接层设计包含两层隐藏层（1024→512→1），并集成Dropout减少过拟合
