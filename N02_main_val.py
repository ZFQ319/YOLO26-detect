import torch

from my_others_code.ModelBenchamrk import ModelBenchmark
from ultralytics import YOLO


def main():

    model_path = 'runs\\obb\\R-SSDD\\yolov8n-obb-SGAM-StrC2f-SD_SSDD-obb_640_MaxEpoch300_SGD_lr0.01_seed42\\weights\\best.pt'
    # data_path = "datasets/HRSID-obb.yaml"

    # 加载yolo网络模型
    model = YOLO(model_path)

    # 选择设备
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 验证模型性能
    model.val(
        # data=data_path,
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.5,
        device=device,
        profile=True,
        save_txt=True,  # 将验证集的预测结果保存在.txt文件中
        save_conf=True  # (bool) 保存信心分数并附带结果
    )

    # 其他性能指标计算
    benchmark = ModelBenchmark(
        model_path=model_path,
        model=model,
        img_size=640,
        batch=1,
        device=device,
        half=False
    )
    benchmark.run(warmup=50, iters=200)

    # 将模型导出为 ONNX 格式
    model.export(
        format="onnx",
        imgsz=640,
        simplify=True,  # 把复杂计算图优化成更简单的图。
        opset=12,  # ONNX 每个版本支持不同算子,版本指定
        dynamic=True,  # 是否动态输入尺寸
    )


if __name__ == '__main__':
    main()
