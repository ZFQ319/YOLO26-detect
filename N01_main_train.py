from ultralytics import YOLO

if __name__ == '__main__':
    seed = 42
    epoch_num = 300
    batch_size = 16
    image_size = 640
    learn_rate = 0.01  # 初始学习率（即SGD=1E-2，亚当=1E-3）
    optimizer = 'SGD'  # 要使用的优化器， choices=['SGD'， 'Adam'， 'AdamW'， 'RMSProp']

    datasets_name = 'SSDD-bbox'
    model_name = 'yolov5n.yaml'
    model_path = f'ultralytics/cfg/models/{'v5'}/{model_name}'
    # model_path = f'my_model/Ref_model/{model_name}'
    datasets_path = f'datasets/{datasets_name}.yaml'
    result_out_path = (f'{model_name[:-5]}{''}_{datasets_name}_{str(image_size)}_'
                       f'MaxEpoch{str(epoch_num)}_{optimizer}_lr{str(learn_rate)}_seed{str(seed)}')

    # 加载yolo网络模型
    model = YOLO(model_path)  # 从头开始创建新的 YOLO 模型

    # 使用自定义数据集训练模型 ,设置超参数
    model.train(
        # model=None,  # 模型文件的路径，即 yolov8.pt，yolov8.yaml
        data=datasets_path,  # 数据文件的路径，即 coco128.yaml
        epochs=epoch_num,  # 要训练的时期数
        save_period=-1,  # 没训练val_period轮次验证一次
        # patience=20,  # 等待没有明显改善以提前停止训练的时期
        batch=batch_size,  # 每批图像数（自动批处理为 -1）
        imgsz=image_size,  # 输入图像的大小为整数或 W，H
        cache=False,  # 真/内存、磁盘 或 假使用缓存加载数据
        device=0,  # 要运行的设备，即 CUDA 设备=0 或设备=0，1，2，3 或设备=CPU
        workers=0,  # 用于数据加载=的工作线程数（如果为 DDP，则按 RANK 计算）
        # project= result_out_path,# 项目根目录
        name= result_out_path,  # 实验名称
        exist_ok=True,  # 是否覆盖现有实验
        optimizer=optimizer,  # 要使用的优化器， choices=['SGD'， 'Adam'， 'AdamW'， 'RMSProp']
        verbose=True,  # 是否打印详细输出
        seed=seed,  # 用于重现性的随机种子
        deterministic=True,  # 是否启用确定性模式
        close_mosaic=30,  # （int） 禁用最终纪元的镶嵌增强
        amp=False,  # 自动混合精度 （AMP） 训练，选择=[真，假]
        lr0=learn_rate,  # 初始学习率（即SGD=1E-2，亚当=1E-3）
        cos_lr=True, # 余弦学习率调整策略
        # # lrf=0.01,  # 最终学习率 （LR0 * LRF）
        momentum=0.937,  # 新加坡元动量/亚当贝塔1
        weight_decay=0.0005,  # 优化器重量衰减 5E-4
        # conf=0.3,
        iou=0.5,
        # box=7.5,  # (float) box损增益
        # cls=0.5,  # (float) 分类损耗增益
        # dfl=1.5,  # (float) 分布焦损增益
        # angle=1,  # （float）定向角度损失增益（OBB任务）
        val=True,  # 在训练期间验证/测试
    )

    # 将模型导出为 ONNX 格式
    model.export(
            format="onnx",
            imgsz=640,
            simplify=True,  # 把复杂计算图优化成更简单的图。
            opset=12,  # ONNX 每个版本支持不同算子,版本指定
            dynamic=True, # 是否动态输入尺寸
    )
