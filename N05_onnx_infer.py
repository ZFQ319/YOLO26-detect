from matplotlib import pyplot as plt

from my_others_code.YOLO_onnx_infer import Infer


def main():
    """
    主函数：程序入口
    功能：配置参数、创建推理器、执行推理、显示结果
    """
    # 指定测试图像的路径（请根据实际情况修改为您的图像路径）
    image_path = 'E:/ZFQ_Projects/Python_Projects/Datasets_detect/SSDD/bbox/images/test/001111.jpg'
    # 指定ONNX模型文件的路径（训练好的YOLOv8 OBB模型）
    model_path = "runs/detect/paper2/SSDD/yolov8n_SSDD-bbox_640_MaxEpoch300_SGD_lr0.01_seed42/weights/best.onnx"
    # 定义类别名称列表（本模型只检测船舶这一类）
    CLASSESE = ['ship']
    # 创建推理器实例，配置为OBB旋转框检测任务
    OBB_infer = Infer(
        model_path=model_path,  # 传入模型文件路径
        task="detect",  # 任务类型：obb表示旋转框检测
        classes=CLASSESE,  # 传入类别列表
        conf_thres=0.25,  # 置信度阈值：低于0.25的检测将被过滤
        iou_thres=0.5,  # NMS的IoU阈值：大于0.5的框会被合并
        imgsz=640,  # 模型输入图像尺寸（640x640）
        device= 'cpu'
    )
    # 执行推理，对指定图像进行检测
    # 返回值格式：[x, y, w, h, angle, confidence, class_id]
    results = OBB_infer.infer(image_path=image_path)
    # 在图像上绘制检测结果并显示（调用类的绘图方法）
    images_with_PreRes = OBB_infer.draw_result(results)
    # 使用matplotlib显示绘制后的图像（RGB格式）
    plt.imshow(images_with_PreRes)
    # 显示图像窗口
    plt.show()

# 程序入口点
if __name__ == '__main__':
    # 调用主函数，启动程序
    main()