import time
from typing import List, Tuple
# 导入OpenCV库，用于图像读取、处理、绘制等计算机视觉操作
import cv2
# 导入numpy库，用于数组运算和数值计算
import numpy as np
# 导入onnxruntime库，用于加载和运行ONNX格式的深度学习模型
import onnxruntime as ort
# 导入PyTorch库，用于张量操作（虽然主要用numpy，但NMS函数需要torch张量）
import torch
# 从ultralytics库导入NMS（非极大值抑制）函数，用于去除冗余检测框
from .nms import non_max_suppression



class Infer:
    """
    YOLO推理类
    功能: 加载ONNX模型、预处理图像、执行推理、后处理、可视化结果
    
    支持两种任务: detect【水平矩形框】和obb【旋转矩形框】
    
    成员函数:
    1. __init__()         - 初始化推理器，加载模型, 外部调用
    2. infer()            - 推理函数, 外部调用
        3. preprocess()            - 预处理函数, infer()中调用
        4. post_process()          - 后处理函数, infer()中调用
            5. map_predictions_to_original()     - 检测结果坐标从预处理后的图像映射回原始图像, post_process()中调用
            
    """

    def __init__(
            self,
            model_path: str,  # ONNX模型文件路径（必需参数）
            task: str = "detect",  # 任务类型，默认"detect"，可选"obb"
            conf_thres: float = 0.25,  # 置信度阈值，默认0.25
            iou_thres: float = 0.45,  # NMS的IoU阈值，默认0.45
            imgsz: int = 640,  # 模型输入图像尺寸（正方形），默认640
            device: str = "cuda"  # 推理设备，默认"cuda"，可选"cpu"
    ):
        """
        初始化推理器，加载模型并配置参数

        Args:
            model_path: 模型路径（字符串）
            task: 检测任务 "obb":旋转矩形定位，"detect":水平矩形定位
            conf_thres: 置信度阈值
            iou_thres: IoU阈值
            imgsz: 输入图像尺寸
            device: 推理设备 ("cuda"或"cpu")
        """

        # 初始化存储处理后图像的变量（将在infer方法中赋值）
        # 获取原始图像的宽度和高度
        self.orig_h, self.orig_w = None, None

        # 初始化图像预处理后的缩放比例和边缘填充
        self.dh = None  # 两侧填充
        self.dw = None  # 上下填充
        self.ratio = None  # 缩放比例
       
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

    # ======================
    # 预处理
    # ======================
    def preprocess(
        self, 
        image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        图像预处理函数
        功能：调整图像大小、填充到正方形、归一化、转换为模型输入格式

        Args:
            image: 原始BGR图像
                - 类型: numpy.ndarray
                - 形状: [H, W, 3]
                - 数据类型: np.uint8
                - 取值范围: 0-255

        Returns:
            Tuple[np.ndarray, float, float, float]:           
                - img: 模型输入张量
                    * 类型: numpy.ndarray
                    * 形状: [1, 3, imgsz, imgsz]
                    * 数据类型: np.float32
                    * 说明: 已归一化到[0, 1]，CHW格式
                    
                - r: 缩放比例
                    * 类型: float
                    * 说明: 原始图像到处理后图像的缩放倍数
                    * 计算公式: min(imgsz / h, imgsz / w)
                    
                - dw: 宽度方向填充量
                    * 类型: float
                    * 说明: 左右各填充的像素数
                    * 单位: 像素
                    
                - dh: 高度方向填充量
                    * 类型: float
                    * 说明: 上下各填充的像素数
                    * 单位: 像素

        Raises:
            ValueError: 
                - 输入图像为None
                - 图像格式不正确,不是3通道
                - 图像尺寸无效
            TypeError: 
                - 输入图像不是numpy数组
                - 输入图像数据类型不正确

        Examples:
            >>> # 基本使用
            >>> img, r, dw, dh = self.preprocess(image)
            >>> print(f"缩放比例: {r:.3f}")
            >>> print(f"填充: 左={dw}, 右={dw}, 上={dh}, 下={dh}")
            >>> 
            >>> # 检查输出形状
            >>> print(f"模型输入形状: {img.shape}")  # (1, 3, 640, 640)
        """
    # 参数验证
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
    def infer(self, image: np.ndarray) -> List[List[float]]:
        """
        推理函数: 加载图像、预处理、执行ONNX推理、后处理
        
        Args:
            image: 原始BGR图像
                - 类型: numpy.ndarray
                - 形状: [H, W, 3]
                - 数据类型: np.uint8
                - 取值范围: 0-255

        Returns:
            List[List[float]]: 检测结果列表，每个检测结果为一个浮点数列表
                
                水平矩形框 (HBB): [x1, y1, x2, y2, confidence, class_id]
                
                | 索引 | 字段名 | 说明 |
                |:----:|:------:|:----:|
                | 0 | x1 | 左上角 x 坐标 |
                | 1 | y1 | 左上角 y 坐标 |
                | 2 | x2 | 右下角 x 坐标 |
                | 3 | y2 | 右下角 y 坐标 |
                | 4 | confidence | 置信度分数 (0.0 - 1.0) |
                | 5 | class_id | 类别 ID |
                
                旋转矩形框 (OBB): [cx, cy, w, h, angle, confidence, class_id]
                
                | 索引 | 字段名 | 说明 |
                |:----:|:------:|:----:|
                | 0 | cx | 中心点 x 坐标 |
                | 1 | cy | 中心点 y 坐标 |
                | 2 | w | 宽度 |
                | 3 | h | 高度 |
                | 4 | angle | 旋转角度 (弧度) |
                | 5 | confidence | 置信度分数 (0.0 - 1.0) |
                | 6 | class_id | 类别 ID |

        Raises:
            ValueError: 
                - 输入图像为None
                - 图像格式不正确, 不是3通道BGR
                - 图像尺寸无效
                - 推理失败或输出为空
            TypeError:
                - 输入图像不是numpy.ndarray
                - 输入图像数据类型不是uint8
            RuntimeError:
                - ONNX推理失败
                - 模型加载失败

        Examples:
            >>> # 基本使用
            >>> results = self.infer(image)
            >>> print(f"检测到 {len(results)} 个目标")
            >>> 
            >>> # 遍历结果（水平框）
            >>> for det in results:
            ...     x1, y1, x2, y2, conf, cls_id = det
            ...     class_name = self.classes[int(cls_id)] if self.classes else str(int(cls_id))
            ...     print(f"{class_name}: {conf:.2f} at ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
            ...
            >>> # 遍历结果（旋转框）
            >>> for det in results:
            ...     cx, cy, w, h, angle, conf, cls_id = det
            ...     class_name = self.classes[int(cls_id)] if self.classes else str(int(cls_id))
            ...     print(f"{class_name}: {conf:.2f} at ({cx:.0f}, {cy:.0f}) size={w:.0f}x{h:.0f} angle={angle:.2f}")
            ...
            >>> # 筛选高置信度结果
            >>> high_conf = [d for d in results if d[4 if len(d) == 6 else 5] > 0.8]
            >>> print(f"高置信度目标: {len(high_conf)} 个")
        """
        start_time = time.time()
       
        # 如果图像读取成功（image不为None）
        if image is not None:
            print(f"开始数据预处理-----", end='')
            load_img_time = time.time()
            # 获取原始图像的宽度和高度
            self.orig_h, self.orig_w = image.shape[:2]
            # 预处理图像，获取处理后的可视化图像、模型输入、缩放比例、填充量
            # input是模型输入张量
            # ratio, dw, dh用于后续坐标逆变换
            input, self.ratio, self.dw, self.dh = self.preprocess(image)

            preprocess_time = time.time()
            print(f'耗时：{(preprocess_time-load_img_time):.4f} ms')

            print(f"开始推理----------", end='')
            # 执行ONNX模型推理
            # session.run返回输出张量列表
            outputs = self.session.run(
                self.output_names,  # 输出节点名称列表
                {self.input_name: input}  # 输入节点名称和输入张量的字典
            )

            infer_time = time.time()
            print(f'耗时：{(infer_time - preprocess_time):.4f} ms')

            print(f"后处理-----------", end='')
            # 对模型输出进行后处理（包括置信度筛选、NMS、坐标格式转换等）
            pre_result = self.post_process(outputs)
            post_process_time = time.time()

            print(f'耗时：{(post_process_time - infer_time):.4f} ms')
            print(f"总耗时：{(post_process_time - start_time):.4f} ms, "
                  f"   FPS: {1000/(post_process_time-start_time+0.00001):.4f}")
            # 返回处理后的结果（坐标已在原图坐标系下）
            return pre_result
        else:
            # 如果图像读取失败（文件不存在或格式不支持），抛出异常
            raise FileNotFoundError(
                f"图像文件不存在或无法读取"
            )

    # ======================
    # 后处理
    # ======================
    def post_process(
            self, 
            result: List[np.ndarray]
            ):
        """
        YOLO输出后处理函数
        功能: 执行置信度筛选、NMS (非极大值抑制)、坐标映射原始图像, 适配OBB/bbox格式

        Args:
            result: 模型原始输出
                - 类型: List[np.ndarray]
                - 说明: ONNX模型推理输出的numpy数组列表
                - 格式: [预测框坐标, 置信度, 类别分数] 或 多尺度输出

        Returns:
            List[float]: NMS处理后的检测结果
            
            水平矩形框 (HBB):
                - 类型: List[float]
                - 形状: [N, 6]
                - 格式: [x1, y1, x2, y2, confidence, class_id]
                
            旋转矩形框 (OBB):
                - 类型: List[float]
                - 形状: [N, 7]
                - 格式: [cx, cy, w, h, angle, confidence, class_id]

        Raises:
            ValueError: 
                - 输入结果为空
                - 输出格式不符合预期
            RuntimeError:
                - NMS执行失败

        Examples:
            >>> # 基本使用
            >>> outputs = session.run(['output'], {input_name: img})
            >>> detections = self.postprocess(outputs)
            >>> print(f"NMS后检测框数量: {len(detections)}")
            >>> 
            >>> # 遍历检测结果
            >>> for det in detections:
            ...     if self.task == "obb":
            ...         cx, cy, w, h, angle, conf, cls_id = det
            ...     else:
            ...         x1, y1, x2, y2, conf, cls_id = det
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
                iou_thres=self.iou_thres,  # IoU阈值（0.3比默认值更严格）
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
                iou_thres=self.iou_thres
            )[0]

        # 处理空结果和批次维度
        if pre is None:
            # 如果没有检测到任何目标，返回空的torch张量
            return []
        else:
            # 张量转为N维数组
            pre = pre.detach().cpu().numpy()
            # 目标坐标、检测框尺度映射回原始图像
            pre = self.map_predictions_to_original(pre)
            return pre

    # ======================
    # 坐标映射回原始图像
    # ======================
    def map_predictions_to_original(
            self, 
            predictions: np.ndarray
        ) -> np.ndarray:
            """
            将预测结果从预处理图像坐标系映射回原始图像坐标系

            Args:
                predictions: 在预处理图像坐标系下的预测结果
                    - 类型: numpy.ndarray
                    - 格式:
                        - HBB任务: [N, 6] -> [x1, y1, x2, y2, conf, cls]
                        - OBB任务: [N, 7] -> [cx, cy, w, h, angle, conf, cls]
                    - 说明: 坐标值在预处理后的正方形图像坐标系下

            Returns:
                mapped_predictions: 在原始图像坐标系下的预测结果
                    - 类型: numpy.ndarray
                    - 格式: 与输入相同，但坐标值已映射回原图
                    - 说明: 坐标值在原始图像坐标系下

            Raises:
                ValueError: 
                    - predictions 为 None 或空
                    - predictions 数据类型不正确
                    - predictions 列数不符合预期 (HBB需要6列，OBB需要7列)
                TypeError:
                    - predictions 不是 numpy.ndarray

            Examples:
                >>> # HBB任务
                >>> preds = np.array([
                ...     [100, 150, 200, 250, 0.9, 0],
                ...     [300, 350, 400, 450, 0.8, 1]
                ... ])
                >>> mapped = self.map_predictions_to_original(preds)
                >>> print(mapped)  # 坐标已映射到原图
                >>>
                >>> # OBB任务
                >>> preds = np.array([
                ...     [150, 200, 80, 60, 0.5, 0.9, 0],
                ...     [350, 400, 100, 80, 0.3, 0.8, 1]
                ... ])
                >>> mapped = self.map_predictions_to_original(preds)
                >>> print(mapped)  # 坐标已映射到原图
            """
            # 创建副本，避免修改原始数据
            mapped_predictions = predictions.copy()
    
            for i in range(len(mapped_predictions)):
                if self.task == 'obb':
                    # OBB任务：映射中心点坐标(x, y)和宽高(w, h)
                    # 1. 映射中心点x坐标：减去左填充量，然后除以缩放比例
                    #    x_original = (x_preprocessed - dw) / ratio
                    mapped_predictions[i, 0] = (predictions[i, 0] - self.dw) / self.ratio
                    # 2. 映射中心点y坐标：减去上填充量，然后除以缩放比例
                    #    y_original = (y_preprocessed - dh) / ratio
                    mapped_predictions[i, 1] = (predictions[i, 1] - self.dh) / self.ratio
                    # 3. 映射宽度：除以缩放比例（宽度缩放与整体缩放一致）
                    #    w_original = w_preprocessed / ratio
                    mapped_predictions[i, 2] = predictions[i, 2] / self.ratio
                    # 4. 映射高度：除以缩放比例（高度缩放与整体缩放一致）
                    #    h_original = h_preprocessed / ratio
                    mapped_predictions[i, 3] = predictions[i, 3] / self.ratio
                    # 5. 角度保持不变（角度不受缩放和填充影响）
                    #    angle_original = angle_preprocessed
                    # 注意：角度值不需要变换，保持不变
    
                    # 可选：添加边界裁剪，确保坐标不超出原始图像范围
                    # 裁剪中心点x坐标到[0, self.orig_w]范围内
                    mapped_predictions[i, 0] = np.clip(mapped_predictions[i, 0], 0, self.orig_w)
                    # 裁剪中心点y坐标到[0, self.orig_h]范围内
                    mapped_predictions[i, 1] = np.clip(mapped_predictions[i, 1], 0, self.orig_h)
                    # 确保宽高为正数
                    mapped_predictions[i, 2] = max(mapped_predictions[i, 2], 1.0)
                    mapped_predictions[i, 3] = max(mapped_predictions[i, 3], 1.0)
    
                else:
                    # Detect任务：映射边界框坐标[x1, y1, x2, y2]
                    # 1. 映射x1坐标
                    mapped_predictions[i, 0] = (predictions[i, 0] - self.dw) / self.ratio
                    # 2. 映射y1坐标
                    mapped_predictions[i, 1] = (predictions[i, 1] - self.dh) / self.ratio
                    # 3. 映射x2坐标
                    mapped_predictions[i, 2] = (predictions[i, 2] - self.dw) / self.ratio
                    # 4. 映射y2坐标
                    mapped_predictions[i, 3] = (predictions[i, 3] - self.dh) / self.ratio
    
                    # 裁剪坐标到原始图像范围内
                    mapped_predictions[i, 0] = np.clip(mapped_predictions[i, 0], 0, self.orig_w)
                    mapped_predictions[i, 1] = np.clip(mapped_predictions[i, 1], 0, self.orig_h)
                    mapped_predictions[i, 2] = np.clip(mapped_predictions[i, 2], 0, self.orig_w)
                    mapped_predictions[i, 3] = np.clip(mapped_predictions[i, 3], 0, self.orig_h)
    
                    # 确保坐标有效（x1 < x2, y1 < y2）
                    if mapped_predictions[i, 0] > mapped_predictions[i, 2]:
                        mapped_predictions[i, 0], mapped_predictions[i, 2] = \
                            mapped_predictions[i, 2], mapped_predictions[i, 0]
                    if mapped_predictions[i, 1] > mapped_predictions[i, 3]:
                        mapped_predictions[i, 1], mapped_predictions[i, 3] = \
                            mapped_predictions[i, 3], mapped_predictions[i, 1]
    
            # 转换回torch张量
            return mapped_predictions
