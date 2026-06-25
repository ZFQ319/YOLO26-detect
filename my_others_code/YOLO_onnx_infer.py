# 导入数学函数模块，用于计算距离、开方等数学运算
import math
import time

# 导入OpenCV库，用于图像读取、处理、绘制等计算机视觉操作
import cv2
# 导入matplotlib库，用于绘图显示
import matplotlib
from torch._C import device

from my_others_code.My_Plot import draw_obb_on_images, draw_bbox_on_image

# 设置matplotlib后端为TkAgg，用于在独立的Tkinter窗口中显示图形,这可以避免在某些环境中（如无GUI的服务器）出现显示问题
matplotlib.use("TkAgg")
# 导入matplotlib的pyplot模块，用于显示图像
import matplotlib.pyplot as plt
# 导入numpy库，用于数组运算和数值计算
import numpy as np
# 导入onnxruntime库，用于加载和运行ONNX格式的深度学习模型
import onnxruntime as ort
# 导入PyTorch库，用于张量操作（虽然主要用numpy，但NMS函数需要torch张量）
import torch
# 从ultralytics库导入NMS（非极大值抑制）函数，用于去除冗余检测框
from ultralytics.utils.nms import non_max_suppression
# 从ultralytics库导入坐标转换函数，将中心点+宽高+角度转换为8个顶点坐标
# 这是OBB（旋转框）检测的关键转换函数
from ultralytics.utils.ops import xywhr2xyxyxyxy


def main():
    """
    主函数：程序入口
    功能：配置参数、创建推理器、执行推理、显示结果
    """
    # 指定测试图像的路径（请根据实际情况修改为您的图像路径）
    image_path = 'E:/ZFQ_Projects/Python_Projects/Datasets_detect/SSDD/obb/images/test/000729.jpg'
    # 指定ONNX模型文件的路径（训练好的YOLOv8 OBB模型）
    model_path = "../runs/obb/R-SSDD/yolov8n-obb-SD_SSDD-obb_640_MaxEpoch300_SGD_lr0.01_seed42/weights/best.onnx"
    # 定义类别名称列表（本模型只检测船舶这一类）
    CLASSESE = ['ship']
    # 创建推理器实例，配置为OBB旋转框检测任务
    OBB_infer = Infer(
        model_path=model_path,  # 传入模型文件路径
        task="obb",  # 任务类型：obb表示旋转框检测
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
    OBB_infer.draw_result(results)


class Infer:
    """
    YOLO推理类
    功能：加载ONNX模型、预处理图像、执行推理、后处理、可视化结果
    支持两种任务：detect（水平矩形框）和obb（旋转矩形框）
    """

    def __init__(
            self,
            model_path,  # ONNX模型文件路径（必需参数）
            task="detect",  # 任务类型，默认"detect"，可选"obb"
            classes=None,  # 类别名称列表，如['ship', 'car']，默认None
            conf_thres=0.25,  # 置信度阈值，默认0.25
            iou_thres=0.45,  # NMS的IoU阈值，默认0.45
            imgsz=640,  # 模型输入图像尺寸（正方形），默认640
            device="cuda"  # 推理设备，默认"cuda"，可选"cpu"
    ):
        """
        初始化推理器，加载模型并配置参数

        Args:
            model_path: 模型路径（字符串）
            task: 检测任务 "obb":旋转矩形定位，"detect":水平矩形定位
            classes: 类别名称列表（List[str]）
            conf_thres: 置信度阈值（float）
            iou_thres: IoU阈值（float）
            imgsz: 输入图像尺寸（int）
            device: 推理设备（"cuda"或"cpu"）
        """

        # 初始化存储处理后图像的变量（将在infer方法中赋值）
        self.original_image = None
        # 初始化图像预处理后的缩放比例和边缘填充
        self.dh = None  # 两侧填充
        self.dw = None  # 上下填充
        self.ratio = None  # 缩放比例
        # 存储类别名称列表
        self.classes = classes if classes is not None else []

        # 存储模型路径
        self.model_path = model_path
        # 存储任务类型（detect或obb）
        self.task = task

        # 存储置信度阈值（用于过滤低置信度检测）
        self.conf_thres = conf_thres
        # 存储IoU阈值（用于NMS去重）
        self.iou_thres = iou_thres

        # 存储输入图像尺寸（正方形边长）
        self.imgsz = imgsz

        # 存储推理设备（cuda或cpu）
        self.device = device

        # 根据设备选择ONNX Runtime的执行提供者（provider）
        # ONNX Runtime支持多种后端加速，如CUDA、CPU、TensorRT等
        providers = (
            # 如果使用CUDA，优先使用CUDA执行提供者，备选CPU
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device.lower() == "cuda"
            # 否则只使用CPU执行提供者
            else ["CPUExecutionProvider"]
        )

        # 创建ONNX Runtime推理会话（InferenceSession）
        # 这一步会加载模型并进行初始化
        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        # 获取模型输入节点的名称（通常为"images"或"input"）
        self.input_name = self.session.get_inputs()[0].name

        # 获取所有输出节点的名称列表（可能有多个输出）
        self.output_names = [
            x.name
            for x in self.session.get_outputs()
        ]

        # 打印当前使用的执行提供者（调试信息）
        print("网络推理设备采用:", self.session.get_providers())

        # 打印模型输入张量的形状（用于调试和验证）
        print(
            "网络模型的输入形状需为:",
            self.session.get_inputs()[0].shape
        )

        # 打印所有输出张量的形状（用于调试和验证）
        print(
            "网络模型的输出形状为:",
            [o.shape for o in self.session.get_outputs()]
        )

        # 方法3：预定义一组深色（适合类别数量较少的情况）
        deep_colors_palette = [
            [255, 0, 0],  # 红色
            [0, 255, 0],  # 绿色
            [0, 0, 255],  # 蓝色
            [0, 255, 255],  # 黄色
            [255, 0, 255],  # 品红
            [255, 255, 0],  # 青色
            [128, 0, 255],  # 橙红
            [0, 128, 255],  # 橙黄
            [255, 0, 128],  # 紫罗兰
            [128, 255, 0],  # 黄绿
            [0, 255, 128],  # 蓝绿
            [255, 128, 0],  # 天蓝
        ]

        # 如果类别数超过预定义颜色，循环使用
        self.colors = []
        for i in range(len(self.classes)):
            self.colors.append(deep_colors_palette[i % len(deep_colors_palette)])

    def preprocess(self, image):
        """
        图像预处理函数
        功能：调整图像大小、填充到正方形、归一化、转换为模型输入格式

        Args:
            image: 原始BGR图像（numpy数组，shape为[H, W, 3]）

        Returns:
            img: 处理后的BGR图像（用于可视化，已填充到正方形）
            no_img: 模型输入张量（shape为[1, 3, imgsz, imgsz]）
            r: 缩放比例（原始图像到处理后的缩放倍数）
            dw: 宽度方向填充量（左右各填充多少像素）
            dh: 高度方向填充量（上下各填充多少像素）
        """
        # 获取原始图像的高度h0和宽度w0
        h0, w0 = image.shape[:2]
        # 计算缩放比例：取两个缩放比例中的较小值
        # 这样可以确保图像能够完整地放入目标尺寸内，不会溢出
        r = min(
            self.imgsz / h0,  # 高度方向的缩放比例
            self.imgsz / w0  # 宽度方向的缩放比例
        )
        # 计算缩放后的宽度（整数）
        nw = int(w0 * r)
        # 计算缩放后的高度（整数）
        nh = int(h0 * r)
        # 使用OpenCV的resize函数缩放图像到目标尺寸
        # 使用默认的插值方法（双线性插值）
        resized = cv2.resize(
            image,
            (nw, nh)
        )
        # 创建正方形画布，尺寸为[imgsz, imgsz, 3]
        # 填充值为114（YOLO系列算法常用的填充值，是ImageNet数据集的均值）
        canvas = np.full(
            (self.imgsz, self.imgsz, 3),
            114,
            dtype=np.uint8
        )
        # 计算宽度方向的填充量（左右对称填充）
        # 使用整数除法确保左右填充量相等
        dw = (self.imgsz - nw) // 2
        # 计算高度方向的填充量（上下对称填充）
        dh = (self.imgsz - nh) // 2
        # 将缩放后的图像放置到画布中央
        # 使用切片操作将resized图像复制到canvas的中央区域
        canvas[
            dh:dh + nh,  # 高度范围
            dw:dw + nw  # 宽度范围
        ] = resized
        # 保存处理后的BGR图像（用于后续可视化绘制）
        img = canvas
        # 将BGR格式转换为RGB格式（因为深度学习模型通常使用RGB格式训练）
        img = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        )
        # 将数据类型从uint8转换为float32（浮点数）
        img = img.astype(np.float32)
        # 归一化到[0, 1]范围：将所有像素值除以255.0
        img /= 255.
        # 转换维度顺序：HWC（高、宽、通道） -> CHW（通道、高、宽）
        # 这是因为PyTorch等深度学习框架使用CHW格式
        img = img.transpose(2, 0, 1)
        # 添加batch维度：(C, H, W) -> (1, C, H, W)
        # 因为模型推理时需要batch维度（即使只有一张图）
        img = np.expand_dims(img, 0)
        # 返回处理后的可视化图像、模型输入、缩放比例、填充量
        return img, r, dw, dh

    # ======================
    # ONNX推理
    # ======================
    def infer(self, image_path):
        """
        推理函数：加载图像、预处理、执行ONNX推理、后处理

        重要改进：后处理完成后，将预测结果坐标从预处理图像坐标系
        映射回原始图像坐标系

        Args:
            image_path: 图像文件路径（字符串）

        Returns:
            pre_result: 后处理后的检测结果（torch.Tensor），坐标已在原图坐标系下
        """
        cost_times=[]
        start_time = time.time()
        print(f"加载图像：{image_path}")

        # 使用OpenCV读取图像（返回BGR格式的numpy数组）
        image = cv2.imread(image_path)
        # 如果图像读取成功（image不为None）
        if image is not None:
            # 保存原始图像（用于最终绘制）
            self.original_image = image.copy()

            load_img_time = time.time()
            print(f"开始数据预处理-----", end='')
            # 预处理图像，获取处理后的可视化图像、模型输入、缩放比例、填充量
            # inp是模型输入张量
            # ratio, dw, dh用于后续坐标逆变换
            inp, self.ratio, self.dw, self.dh = self.preprocess(image)

            preprocess_time = time.time()
            print(f'耗时：{(preprocess_time-load_img_time):.4f} ms')

            print(f"开始推理----------", end='')
            # 执行ONNX模型推理
            # session.run返回输出张量列表
            outputs = self.session.run(
                self.output_names,  # 输出节点名称列表
                {self.input_name: inp}  # 输入节点名称和输入张量的字典
            )

            infer_time = time.time()
            print(f'耗时：{(infer_time - preprocess_time):.4f} ms')

            print(f"后处理-----------", end='')
            # 对模型输出进行后处理（包括NMS、坐标格式转换等）
            # 此时的预测结果坐标仍在预处理图像坐标系（640x640带填充）
            pre_result = self.post_process(outputs)

            # ========== 关键改进：将预测结果映射回原图坐标 ==========
            if pre_result is not None and len(pre_result) > 0:
                pre_result = self.map_predictions_to_original(pre_result)

            post_process_time = time.time()
            print(f'耗时：{(post_process_time - infer_time):.4f} ms')
            print(f"总耗时：{(post_process_time - start_time):.4f} ms, "
                  f"   FPS: {1000/(post_process_time-start_time):.4f}")
            # 返回处理后的结果（坐标已在原图坐标系下）
            return pre_result
        else:
            # 如果图像读取失败（文件不存在或格式不支持），抛出异常
            raise FileNotFoundError(
                f"读取失败，检查路径是否存在: {image_path}"
            )

    def map_predictions_to_original(self, predictions):
        """
        将预测结果从预处理图像坐标系映射回原始图像坐标系

        Args:
            predictions: 在预处理图像坐标系下的预测结果（torch.Tensor）
                        格式：[N, 7] 其中7列为 [x, y, w, h, angle, conf, cls]
                        对于obb任务，x,y是中心点坐标，w,h是宽高

        Returns:
            mapped_predictions: 在原始图像坐标系下的预测结果（torch.Tensor）
                               格式保持不变，但坐标值已映射回原图
        """
        # 将张量转换为numpy数组进行处理
        predictions_np = predictions.detach().cpu().numpy()
        # 创建副本，避免修改原始数据
        mapped_predictions = predictions_np.copy()

        # 获取原始图像的宽度和高度
        orig_h, orig_w = self.original_image.shape[:2]

        for i in range(len(mapped_predictions)):
            if self.task == 'obb':
                # OBB任务：映射中心点坐标(x, y)和宽高(w, h)
                # 1. 映射中心点x坐标：减去左填充量，然后除以缩放比例
                #    x_original = (x_preprocessed - dw) / ratio
                mapped_predictions[i, 0] = (predictions_np[i, 0] - self.dw) / self.ratio
                # 2. 映射中心点y坐标：减去上填充量，然后除以缩放比例
                #    y_original = (y_preprocessed - dh) / ratio
                mapped_predictions[i, 1] = (predictions_np[i, 1] - self.dh) / self.ratio
                # 3. 映射宽度：除以缩放比例（宽度缩放与整体缩放一致）
                #    w_original = w_preprocessed / ratio
                mapped_predictions[i, 2] = predictions_np[i, 2] / self.ratio
                # 4. 映射高度：除以缩放比例（高度缩放与整体缩放一致）
                #    h_original = h_preprocessed / ratio
                mapped_predictions[i, 3] = predictions_np[i, 3] / self.ratio
                # 5. 角度保持不变（角度不受缩放和填充影响）
                #    angle_original = angle_preprocessed
                # 注意：角度值不需要变换，保持不变

                # 可选：添加边界裁剪，确保坐标不超出原始图像范围
                # 裁剪中心点x坐标到[0, orig_w]范围内
                mapped_predictions[i, 0] = np.clip(mapped_predictions[i, 0], 0, orig_w)
                # 裁剪中心点y坐标到[0, orig_h]范围内
                mapped_predictions[i, 1] = np.clip(mapped_predictions[i, 1], 0, orig_h)
                # 确保宽高为正数
                mapped_predictions[i, 2] = max(mapped_predictions[i, 2], 1.0)
                mapped_predictions[i, 3] = max(mapped_predictions[i, 3], 1.0)

            else:
                # Detect任务：映射边界框坐标[x1, y1, x2, y2]
                # 1. 映射x1坐标
                mapped_predictions[i, 0] = (predictions_np[i, 0] - self.dw) / self.ratio
                # 2. 映射y1坐标
                mapped_predictions[i, 1] = (predictions_np[i, 1] - self.dh) / self.ratio
                # 3. 映射x2坐标
                mapped_predictions[i, 2] = (predictions_np[i, 2] - self.dw) / self.ratio
                # 4. 映射y2坐标
                mapped_predictions[i, 3] = (predictions_np[i, 3] - self.dh) / self.ratio

                # 裁剪坐标到原始图像范围内
                mapped_predictions[i, 0] = np.clip(mapped_predictions[i, 0], 0, orig_w)
                mapped_predictions[i, 1] = np.clip(mapped_predictions[i, 1], 0, orig_h)
                mapped_predictions[i, 2] = np.clip(mapped_predictions[i, 2], 0, orig_w)
                mapped_predictions[i, 3] = np.clip(mapped_predictions[i, 3], 0, orig_h)

                # 确保坐标有效（x1 < x2, y1 < y2）
                if mapped_predictions[i, 0] > mapped_predictions[i, 2]:
                    mapped_predictions[i, 0], mapped_predictions[i, 2] = \
                        mapped_predictions[i, 2], mapped_predictions[i, 0]
                if mapped_predictions[i, 1] > mapped_predictions[i, 3]:
                    mapped_predictions[i, 1], mapped_predictions[i, 3] = \
                        mapped_predictions[i, 3], mapped_predictions[i, 1]

        # 转换回torch张量
        return torch.from_numpy(mapped_predictions)


    # ======================
    # 后处理统一入口
    # ======================
    def post_process(self, result):
        """
        YOLO输出后处理函数
        功能：执行NMS（非极大值抑制），适配OBB/bbox格式

        Args:
            result: 模型原始输出（通常是numpy数组的列表）

        Returns:
            pre: NMS处理后的检测结果（torch.Tensor）
        """
        # 处理模型输出为列表的情况（ONNX Runtime通常返回列表）
        if isinstance(result, list):
            # 如果是列表，取第一个元素并转换为torch张量
            # result[0]是模型的主要输出张量
            result = torch.from_numpy(result[0])

        # 根据任务类型选择不同的NMS处理方式
        if self.task == 'obb':
            # OBB（旋转框）的NMS处理
            # non_max_suppression是Ultralytics提供的NMS函数
            pre = non_max_suppression(
                result,  # 模型输出张量
                conf_thres=self.conf_thres,  # 置信度阈值
                nc=1,  # 类别数量（写死为1，因为只有ship一类）
                iou_thres=0.3,  # IoU阈值（0.3比默认值更严格）
                rotated=True  # 启用旋转框NMS模式
            )[0]  # [0]表示取第一个batch的结果（因为batch size=1）
            # 重新排列输出列的顺序：调整角度和置信度/类别的位置
            # 原始顺序假设为[x, y, w, h, conf, angle, cls]或其他顺序
            # 通过idx调整为[x, y, w, h, angle, conf, cls]的统一格式
            idx = [0, 1, 2, 3, 6, 4, 5]  # 新顺序：[x,y,w,h,角度,置信度,类别]
            pre = pre[:, idx]  # 应用列重排
        else:
            # BBOX（水平框）的NMS处理
            pre = non_max_suppression(
                result,
                conf_thres=self.conf_thres,
                iou_thres=0.65
            )[0]

        # 处理空结果和批次维度
        if pre is None:
            # 如果没有检测到任何目标，返回空的torch张量
            return torch.tensor([])
        # 如果结果张量的维度大于2，去掉batch维度（squeeze操作）
        # 正常情况下pre的shape为[N, 7]（N个检测框，每框7个值）
        pre = pre.squeeze(0) if len(pre.shape) > 2 else pre

        return pre

    def draw_result(
            self,
            prediction
    ):
        """
        绘制检测结果函数
        功能：在图像上绘制检测框、类别标签和置信度

        Args:
            prediction: Tensor或List，预测结果（NMS后的结果）
        """
        # 在特在可视化热力图上绘制检测框、类别、置信度
        # 检查预测结果是否存在且非空
        if prediction is not None and len(prediction) > 0:
            # 遍历每个检测结果（每个检测框）
            for data in prediction:
                # 将torch张量转换为numpy数组并移到CPU（如果之前在GPU上）
                data_np = data.detach().cpu().numpy()
                # 提取类别ID和置信度
                if self.task == 'obb':
                    # OBB任务：提取类别ID（第7个元素，索引6）
                    cls_id = int(data_np[6])
                    # 提取置信度（第6个元素，索引5）
                    conf = float(data_np[5])
                    # 将中心点坐标+宽高+角度转换为8个顶点坐标
                    # xywhr2xyxyxyxy函数将[cx, cy, w, h, angle]转为8个坐标值
                    boxes = xywhr2xyxyxyxy(data_np[:5])
                else:
                    # 水平框任务（这里代码逻辑与OBB不太一致，但保持原样）
                    # 提取类别ID（从第5个元素开始取argmax，即取最高分类别）
                    cls_id = int(data_np[4:].argmax())
                    # 提取最大置信度（类别分数中的最大值）
                    conf = float(data_np[4:].max())
                    # 提取边界框坐标（前4个元素：[x1, y1, x2, y2]）
                    boxes = data_np[:4]

                # 构建标签文本：类别名 + 置信度百分比（取整，不显示小数）
                label = f'{self.classes[cls_id]} {conf * 100:.0f}'

                # 根据任务类型调用不同的绘制函数
                if self.task == 'obb':
                    # 绘制旋转框（调用draw_obb方法）
                    self.original_image = draw_obb_on_images(self.original_image, boxes, self.colors[cls_id], label)
                else:
                    # 绘制水平框（调用draw_bbox方法）
                    self.original_image = draw_bbox_on_image(self.original_image, boxes, self.colors[cls_id], label, 1)

        return  self.original_image

# 程序入口点
if __name__ == '__main__':
    # 调用主函数，启动程序
    main()