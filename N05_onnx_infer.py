import os
import cv2
from utils.Infer import Infer
from utils.draw_image import draw_results


def main():
    """
    主函数：程序入口
    功能：配置参数、创建推理器、执行推理、显示结果
    """
    # 指定测试图像的路径（请根据实际情况修改为您的图像路径）
    image_path = 'E:/ZFQ_Projects/Python_Projects/Datasets_detect/SSDD/obb/images/test'
    # 指定ONNX模型文件的路径（训练好的YOLOv8 OBB模型）
    model_path = "./runs/obb/R-SSDD/yolov8n-obb_SSDD-obb_640_MaxEpoch300_SGD_lr0.01_seed42/weights/best.onnx"
    # 定义类别名称列表（本模型只检测船舶这一类）
    CLASSESE = ['ship']
    # 可选：定义颜色列表，用于绘制不同类别的框（此处未使用）
    colors = [
        [0, 255, 0],  # 绿色
        [0, 0, 255],  # 红色
        [255, 0, 0],  # 蓝色
        [0, 255, 255],  # 黄色
        [255, 0, 255],  # 紫色
        [255, 255, 0]   # 青色 
    ]  

    # 创建推理器实例，配置为OBB旋转框检测任务
    HBB_infer = Infer(
        model_path=model_path,  # 传入模型文件路径
        task="obb",  # 任务类型：detect表示水平框检测
        conf_thres=0.5,  # 置信度阈值：低于0.25的检测将被过滤
        iou_thres=0.5,  # NMS的IoU阈值：大于0.5的框会被合并
        imgsz=640,  # 模型输入图像尺寸（640x640）
        device= 'cuda'  # 指定运行设备为CPU（可选：'cuda'表示GPU）
    )

    images_name = os.listdir(image_path)  # 获取指定路径下的所有图像文件名列表
    for image_name in images_name:
        image = cv2.imread(os.path.join(image_path, image_name))
        if image is None:
            print(f"无法读取图像: {image_name}")
            continue
        # 执行推理，获取检测结果,检测结果为一个列表，每个元素是一个检测框的预测信息
        # 【x1, y1, x2, y2, conf, cls_id】或【cx, cy, w, h, angle, conf, cls_id】
        results = HBB_infer.infer(image=image)
        # 在图像上绘制检测结果并显示（调用类的绘图方法）
        draw_results(
            image=image, 
            prediction=results, 
            classes=CLASSESE, 
            colors=colors, 
            is_obb=True,
            show_confidence = True,
            thickness = 1,
            font_size_alpha = 1
            )
        # 使用matplotlib显示绘制后的图像（RGB格式）
        cv2.imshow('Image', image)  # 显示图像窗口
        # 显示图像窗口
        cv2.waitKey(0)  # 等待按键事件，按任意键关闭窗口
    cv2.destroyAllWindows()

# 程序入口点
if __name__ == '__main__':
    # 调用主函数，启动程序
    main()