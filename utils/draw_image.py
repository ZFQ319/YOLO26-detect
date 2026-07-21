import math
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from .nms import xywhr2xyxyxyxy


def draw_ground_truth_on_images(original_image, label_path):

    """在原始图像上绘制标签"""
    original_image = np.array(original_image).copy()
    labels = open(label_path, 'r')  # 读取标签
    # 根据标签在原图上绘制位置框并标出目标类型（循环获取多个标签并标注）
    for line in labels.readlines():
        line = line.strip('\n')  # 获取当前目标标签
        label = line.split(' ')  # 转换目标标签格式
        if len(label) == 5:
            top_left_x = int((float(label[1]) - float(label[3]) / 2.0) * original_image.shape[1])  # 获取目标左上角坐标x
            top_left_y = int((float(label[2]) - float(label[4]) / 2.0) * original_image.shape[0])  # 获取目标左上角坐标y
            width = int(float(label[3]) * original_image.shape[1])  # 获取目标位置框宽
            height = int(float(label[4]) * original_image.shape[0])  # 获取目标位置框高
            cv2.rectangle(original_image, (top_left_x, top_left_y), (top_left_x + width, top_left_y + height),
                          (0, 255, 0),
                          1)  # 在原图增加位置框
        else:
            new_box = [
                [int(float(label[1]) * original_image.shape[1]), int(float(label[2]) * original_image.shape[0])],
                [int(float(label[3]) * original_image.shape[1]), int(float(label[4]) * original_image.shape[0])],
                [int(float(label[5]) * original_image.shape[1]), int(float(label[6]) * original_image.shape[0])],
                [int(float(label[7]) * original_image.shape[1]), int(float(label[8]) * original_image.shape[0])]]
            point = np.array(new_box).astype(int)
            color = (0, 255, 0)
            thickness = 1
            cv2.line(original_image, point[0], point[1], color, thickness)
            cv2.line(original_image, point[1], point[2], color, thickness)
            cv2.line(original_image, point[2], point[3], color, thickness)
            cv2.line(original_image, point[3], point[0], color, thickness)
    return original_image



def draw_bbox_on_image(
    img: np.ndarray,
    box: Union[List[float], Tuple[float, float, float, float], np.ndarray],
    color: Union[List[int], Tuple[int, int, int]],
    label: str,
    label_font_size_alpha: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """
    在图像上绘制HBB边框和标签信息（水平矩形框）

    Args:
        img: 图像
            - 类型: numpy.ndarray
            - 形状: [H, W, 3]
            - 数据类型: uint8
            - 说明: 会被直接修改（原地绘制）
        
        box: 边界框四个坐标值
            - 类型: Union[List[float], Tuple[float, float, float, float], np.ndarray]
            - 格式: [x1, y1, x2, y2]
            - 说明: 可以是浮点数或整数，会自动转换为整数
        
        color: 框线颜色
            - 类型: Union[List[int], Tuple[int, int, int]]
            - 格式: [B, G, R] (BGR格式)
            - 取值范围: 0-255
            - 示例: [255, 0, 0] 表示红色
        
        label: 标签内容
            - 类型: str
            - 说明: 显示在边框上方的文本，通常为 "类别名 置信度"
        
        label_font_size_alpha: 类别标签和置信度的字号调整系数
            - 类型: float
            - 默认值: 1.0
            - 说明: >1 放大字体，<1 缩小字体
        
        thickness: 边框线条粗细
            - 类型: int
            - 默认值: 2
            - 说明: 边框的像素宽度

    Raises:
        ValueError: 
            - 输入图像为None
            - 图像格式不正确
            - box坐标无效
        TypeError:
            - 输入图像不是numpy.ndarray

    Examples:
        >>> # 基本使用
        >>> image = cv2.imread('test.jpg')
        >>> box = [100, 200, 300, 400]
        >>> color = [0, 255, 0]  # 绿色
        >>> label = "car: 0.95"
        >>> draw_bbox_on_image(image, box, color, label)
        >>>
        >>> # 调整字体大小
        >>> draw_bbox_on_image(image, box, color, label, label_font_size_alpha=1.5)

    Note:
        - 字体大小根据边界框尺寸自适应
        - 坐标会自动转换为整数
        - 使用抗锯齿绘制（LINE_AA）
    """

    # ========== 1. 参数验证 ==========
    if img is None:
        raise ValueError("输入图像为 None")
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入图像必须是 numpy.ndarray，当前类型: {type(img)}")
    
    if img.size == 0:
        raise ValueError("输入图像为空")
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"输入图像必须是3通道BGR图像，当前形状: {img.shape}")
    
    if box is None or len(box) < 4:
        raise ValueError(f"边界框坐标无效: {box}")

    # ========== 2. 提取和转换坐标 ==========
    # 将坐标转换为整数类型（OpenCV的坐标必须是整数）
    x1, y1, x2, y2 = list(map(int, box[:4]))
    
    # 限制坐标在图像范围内
    height, width = img.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    
    # 验证坐标有效性
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"无效的边界框坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    # ========== 3. 验证颜色 ==========
    if color is None or len(color) < 3:
        color = (0, 255, 0)  # 默认绿色
    else:
        color = tuple(color[:3])

    # ========== 4. 绘制矩形框 ==========
    # 参数：图像、左上角坐标、右下角坐标、颜色、线宽、线型
    cv2.rectangle(
        img, 
        (x1, y1), 
        (x2, y2), 
        color=color, 
        thickness=thickness, 
        lineType=cv2.LINE_AA
    )

    # ========== 5. 计算自适应字体大小 ==========
    if label:
        # 计算边界框对角线长度
        bbox_diag = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        
        # 计算图像对角线长度
        img_diag = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
        
        # 计算边界框占图像的比例
        ratio = bbox_diag / img_diag if img_diag > 0 else 0.1
        
        # 字体大小计算公式：50 * ratio^2 + 0.4
        # 经过调整，使字体大小与边界框尺寸成比例
        font_size = label_font_size_alpha * (50 * ratio ** 2 + 0.4)
        
        # 限制字体大小范围
        font_size = max(0.3, min(font_size, 2.0))
        
        # ========== 6. 绘制标签 ==========
        # 调用show_label_on_image方法绘制标签
        show_label_on_image(img, label, x1, y1, font_size, color)



def draw_obb_on_images(
    img: np.ndarray,
    box_8p: Union[List[float], Tuple[float, ...], np.ndarray],
    color: Union[List[int], Tuple[int, int, int]],
    label: str,
    label_font_size_alpha: float = 1.0,
    thickness: int = 2
) -> np.ndarray:
    """
    绘制OBB边框和标签信息（旋转矩形框）

    Args:
        img: 图像
            - 类型: numpy.ndarray
            - 形状: [H, W, 3]
            - 数据类型: uint8
            - 说明: 会被直接修改（原地绘制）
        
        box_8p: OBB框8个顶点坐标
            - 类型: Union[List[float], Tuple[float, ...], np.ndarray]
            - 格式: [x1, y1, x2, y2, x3, y3, x4, y4]
            - 说明: 4个点的x,y坐标连续排列，共8个值
        
        color: 框线颜色
            - 类型: Union[List[int], Tuple[int, int, int]]
            - 格式: [B, G, R] (BGR格式)
            - 取值范围: 0-255
            - 示例: [255, 0, 0] 表示红色
        
        label: 标签内容
            - 类型: str
            - 说明: 显示在旋转框上方的文本，通常为 "类别名 置信度"
        
        label_font_size_alpha: 类别标签和置信度的字号调整系数
            - 类型: float
            - 默认值: 1.0
            - 说明: >1 放大字体，<1 缩小字体
        
        thickness: 边框线条粗细
            - 类型: int
            - 默认值: 2
            - 说明: 边框的像素宽度

    Raises:
        ValueError: 
            - 输入图像为None
            - 图像格式不正确
            - box_8p长度不是8
            - 多边形顶点无效
        TypeError:
            - 输入图像不是numpy.ndarray

    Examples:
        >>> # 基本使用
        >>> image = cv2.imread('test.jpg')
        >>> # OBB 8个顶点坐标: [x1,y1,x2,y2,x3,y3,x4,y4]
        >>> box_8p = [100, 100, 200, 80, 250, 150, 150, 170]
        >>> color = [0, 255, 0]  # 绿色
        >>> label = "ship: 0.95"
        >>> draw_obb_on_images(image, box_8p, color, label)
        >>>
        >>> # 调整字体大小
        >>> draw_obb_on_images(image, box_8p, color, label, label_font_size_alpha=1.5)

    Note:
        - 顶点顺序应为顺时针或逆时针方向
        - 标签会显示在第一个顶点位置
        - 字体大小根据旋转框尺寸自适应
        - 使用抗锯齿绘制（LINE_AA）
    """

    # ========== 1. 参数验证 ==========
    if img is None:
        raise ValueError("输入图像为 None")
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入图像必须是 numpy.ndarray，当前类型: {type(img)}")
    
    if img.size == 0:
        raise ValueError("输入图像为空")
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"输入图像必须是3通道BGR图像，当前形状: {img.shape}")
    
    if box_8p is None:
        raise ValueError("box_8p 为 None")
    
    if box_8p.size != 8:
        raise ValueError(f"box_8p 必须包含8个坐标值，当前长度: {len(box_8p)}")

    # 将8个坐标值重塑为4个顶点的坐标（4x2矩阵）
    # 例如：[x1,y1,x2,y2,x3,y3,x4,y4] -> [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    poly = box_8p.reshape(4, 2).astype(np.int32)
    # 绘制多边形轮廓（旋转框）
    # polylines可以绘制多边形，isClosed=True表示闭合
    cv2.polylines(img, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    # 获取第一个顶点的坐标（用于放置标签文本）
    x, y = poly[0]

    # 计算边框和图像的对角线长
    obb_l = math.sqrt((poly[0][0] - poly[2][0]) ** 2 + (poly[0][1] - poly[2][1]) ** 2)
    img_l = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    # 计算边长占图像高度的比例
    a = obb_l / img_l
    # 字体大小计算公式：a^2 + 0.4
    font_size =label_font_size_alpha* a ** 2 + 0.4
    # 绘制类别标签（调用plot_label方法）
    show_label_on_image(img=img, label=label, x=x, y=y, font_size=font_size, color=color)


def show_label_on_image(
    img: np.ndarray,
    label: str,
    x: int,
    y: int,
    font_size: float = 1.0,
    color: Optional[Union[List[int], Tuple[int, int, int]]] = None,
    alpha: float = 0.3,
    thickness: int = 1,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
) -> np.ndarray:
    """
    绘制标签函数
    功能：绘制带半透明背景的文本标签，文字自适应调整位置

    Args:
        img: 图像
            - 类型: numpy.ndarray
            - 形状: [H, W, 3]
            - 数据类型: uint8
            - 说明: 会被直接修改（原地绘制）
        
        label: 标签内容
            - 类型: str
            - 说明: 要显示的文本内容
        
        x: 标签显示位置的x坐标
            - 类型: int
            - 说明: 文本左上角x坐标
        
        y: 标签显示位置的y坐标
            - 类型: int
            - 说明: 文本顶部y坐标
        
        font_size: 字号大小
            - 类型: float
            - 默认值: 1.0
            - 说明: 字体缩放因子
        
        color: 背景矩形的颜色
            - 类型: Optional[Union[List[int], Tuple[int, int, int]]]
            - 默认值: None (使用红色 [0, 0, 255])
            - 格式: [B, G, R] (BGR格式)
            - 取值范围: 0-255
        
        alpha: 背景透明度
            - 类型: float
            - 默认值: 0.3
            - 范围: 0-1
            - 说明: 0表示完全透明，1表示完全不透明
        
        thickness: 文字线条粗细
            - 类型: int
            - 默认值: 1
            - 说明: 文字笔画宽度，-1表示填充
        
        font_face: 字体类型
            - 类型: int
            - 默认值: cv2.FONT_HERSHEY_SIMPLEX
            - 可选值: 
                - cv2.FONT_HERSHEY_SIMPLEX
                - cv2.FONT_HERSHEY_PLAIN
                - cv2.FONT_HERSHEY_DUPLEX
                - cv2.FONT_HERSHEY_COMPLEX
                - cv2.FONT_HERSHEY_TRIPLEX
                - cv2.FONT_HERSHEY_COMPLEX_SMALL
                - cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
                - cv2.FONT_HERSHEY_SCRIPT_COMPLEX

    Returns:
        img: 绘制后的图像
            - 类型: numpy.ndarray
            - 说明: 与输入是同一个对象（原地修改）

    Raises:
        ValueError: 
            - 输入图像为None
            - 图像格式不正确
            - alpha不在0-1范围内
            - label为空
        TypeError:
            - 输入图像不是numpy.ndarray

    Examples:
        >>> # 基本使用
        >>> image = cv2.imread('test.jpg')
        >>> show_label_on_image(image, "car: 0.95", 100, 200, font_size=0.8)
        >>>
        >>> # 自定义颜色和透明度
        >>> show_label_on_image(
        ...     image, 
        ...     "person: 0.87", 
        ...     100, 200, 
        ...     font_size=1.0,
        ...     color=[0, 255, 0],  # 绿色背景
        ...     alpha=0.5
        ... )
        >>>
        >>> # 粗体文字
        >>> show_label_on_image(
        ...     image, 
        ...     "ship: 0.95", 
        ...     100, 200, 
        ...     font_size=1.2,
        ...     thickness=2
        ... )

    Note:
        - 标签会自动调整位置，避免超出图像边界
        - 使用半透明背景提高可读性
        - 文字颜色固定为白色，确保清晰可见
        - 背景颜色默认为红色 (0, 0, 255)
    """

    # ========== 1. 参数验证 ==========
    if img is None:
        raise ValueError("输入图像为 None")
    
    if not isinstance(img, np.ndarray):
        raise TypeError(f"输入图像必须是 numpy.ndarray，当前类型: {type(img)}")
    
    if img.size == 0:
        raise ValueError("输入图像为空")
    
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError(f"输入图像必须是3通道BGR图像，当前形状: {img.shape}")
    
    if not label:
        raise ValueError("标签内容不能为空")
    
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在0-1之间，当前值: {alpha}")
    # 如果没有指定颜色，默认使用红色[B,G,R]=[0,0,255]？实际是[255,0,0]表示红色
    # 注意：OpenCV使用BGR格式，[255,0,0]表示蓝色，[0,0,255]表示红色
    # 这里注释说红色，但实际[255,0,0]是蓝色，可能是笔误
    if color is None:
        color = [255, 0, 0]
    # 复制原图像，用于创建叠加层（实现半透明效果）
    # 这样可以避免直接修改原图，先在半透明层上绘制背景，再融合
    overlay = img.copy()
    # --- 2. 计算文字尺寸 ---
    # 获取文本的宽度、高度和基线（baseline是文字底部到基线的距离）
    # getTextSize返回(宽度, 高度)和基线
    (text_w, text_h), _ = cv2.getTextSize(
        label,  # 文本内容
        cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型（简单无衬线字体）
        font_size,  # 字体大小
        1  # 字体粗细
    )

    # --- 3. 画背景矩形（在 overlay 上）---
    # 在文本位置绘制半透明背景矩形
    # 矩形坐标：左上角(x, y-text_h-5)，右下角(x+text_w, y)
    # y-text_h-5：向上偏移文本高度+5像素，给文字留出空间
    cv2.rectangle(
        overlay,
        (x, y - text_h - 5),  # 矩形左上角坐标
        (x + text_w, y),  # 矩形右下角坐标
        color,  # 使用类别颜色填充
        -1  # thickness=-1表示填充整个矩形
    )

    # --- 4. 透明融合 ---
    # 将overlay（带背景矩形）与原图按alpha比例融合，实现半透明效果
    # dst = src1 * alpha + src2 * (1 - alpha)
    # 这里overlay作为src1，权重alpha；原图img作为src2，权重1-alpha
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # --- 5. 再画文字（不透明更清晰）---
    # 在半透明背景上绘制白色文字（不透明，保证文字清晰可读）
    cv2.putText(
        img,
        label,  # 文本内容
        (x, y - 5),  # 文本位置（略微抬高5像素，与背景对齐）
        cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
        font_size,  # 字体大小
        (255, 255, 255),  # 白色文字（BGR格式）
        1,  # 字体粗细
        cv2.LINE_AA  # 抗锯齿线型
    )


def draw_results(
    image: np.ndarray,
    prediction: Union[np.ndarray, List],
    classes: List[str],
    colors: List[Tuple[int, int, int]],
    is_obb: bool = False,
    show_confidence: bool = True,
    thickness: int = 2,
    font_size_alpha: float = 1.0
) -> np.ndarray:
    
    """
    绘制检测结果函数
    功能：在图像上绘制检测框、类别标签和置信度

    Args:
        image: 原始图像
            - 类型: numpy.ndarray
            - 形状: [H, W, 3]
            - 数据类型: uint8
            - 说明: 图像会被直接修改（原地绘制）
        
        prediction: 预测结果（NMS后的结果）
            - 类型: Union[np.ndarray, torch.Tensor, List]
            - 格式:
                - HBB任务: [N, 6] -> [x1, y1, x2, y2, confidence, class_id]
                - OBB任务: [N, 7] -> [cx, cy, w, h, angle, confidence, class_id]
            - 说明: 支持numpy数组、PyTorch张量或列表
        
        classes: 类别名称列表
            - 类型: List[str]
            - 说明: 按类别ID索引，如 ['car', 'person', 'bicycle']
        
        colors: 颜色列表
            - 类型: List[Tuple[int, int, int]]
            - 格式: (B, G, R) BGR格式
            - 说明: 按类别ID索引
        
        is_obb: 是否为OBB任务
            - 类型: bool
            - 默认值: False
            - 说明: True表示旋转框，False表示水平框
        
        show_confidence: 是否显示置信度
            - 类型: bool
            - 默认值: True
            - 说明: True显示 "类别: 置信度%"，False只显示类别
        
        thickness: 边框线条粗细
            - 类型: int
            - 默认值: 2
            - 说明: 边界框的线条宽度
        
        font_size_alpha: 字体大小调整系数
            - 类型: float
            - 默认值: 1.0
            - 说明: >1 放大字体，<1 缩小字体

    Raises:
        ValueError: 
            - 输入图像为None
            - 预测结果为空
            - 类别列表为空
            - 颜色列表为空
        TypeError:
            - 输入图像不是numpy.ndarray
            - 预测结果类型不支持

    Examples:
        >>> # HBB任务
        >>> image = cv2.imread('test.jpg')
        >>> detections = np.array([
        ...     [100, 200, 300, 400, 0.95, 0],
        ...     [400, 150, 550, 350, 0.87, 1]
        ... ])
        >>> classes = ['car', 'person']
        >>> colors = [(0, 255, 0), (255, 0, 0)]
        >>> result = draw_results(image, detections, classes, colors)
        >>>
        >>> # OBB任务
        >>> detections_obb = np.array([
        ...     [300, 250, 150, 80, 0.3, 0.95, 0],
        ...     [500, 400, 120, 60, 0.8, 0.87, 1]
        ... ])
        >>> result = draw_results(image, detections_obb, classes, colors, is_obb=True)
    """
    # ========== 1. 参数验证 ==========
    if image is None:
        raise ValueError("输入图像为 None")
    
    if not isinstance(image, np.ndarray):
        raise TypeError(f"输入图像必须是 numpy.ndarray，当前类型: {type(image)}")
    
    if image.size == 0:
        raise ValueError("输入图像为空")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"输入图像必须是3通道BGR图像，当前形状: {image.shape}")
    
    if prediction is None:
        raise ValueError("预测结果为 None")
    
    if len(prediction) == 0:
        return image  # 无检测结果，直接返回原图
    
    if not classes:
        raise ValueError("类别列表不能为空")
    
    if not colors:
        raise ValueError("颜色列表不能为空")

    # ========== 2. 转换预测结果为numpy数组 ==========
    # 处理列表
    if isinstance(prediction, (list, tuple)):
        prediction = np.array(prediction)
    
    # 检查是否为numpy数组
    if not isinstance(prediction, np.ndarray):
        raise TypeError(f"预测结果类型不支持: {type(prediction)}")

    # ========== 3. 遍历每个检测结果 ==========
    for data_np in prediction:
        try:
            # 提取类别ID和置信度
            if is_obb:
                # OBB任务: [cx, cy, w, h, angle, confidence, class_id]
                # 提取类别ID（第7个元素，索引6）
                cls_id = int(data_np[6])
                # 提取置信度（第6个元素，索引5）
                conf = float(data_np[5])
                # 将中心点坐标+宽高+角度转换为8个顶点坐标
                # xywhr2xyxyxyxy函数将[cx, cy, w, h, angle]转为8个坐标值
                boxes = xywhr2xyxyxyxy(data_np[:5])
                
                # 构建标签文本：类别名 + 置信度百分比
                if show_confidence:
                    label = f'{classes[cls_id]} {conf * 100:.0f}%'
                else:
                    label = f'{classes[cls_id]}'
                
                # 绘制旋转框
                draw_obb_on_images(
                    image, 
                    boxes, 
                    colors[cls_id % len(colors)], 
                    label, 
                    font_size_alpha,
                    thickness
                )
                
            else:
                # HBB任务: [x1, y1, x2, y2, confidence, class_id]
                # 提取类别ID（第6个元素，索引5）
                cls_id = int(data_np[5])
                # 提取置信度（第5个元素，索引4）
                conf = float(data_np[4])
                # 提取边界框坐标（前4个元素：[x1, y1, x2, y2]）
                boxes = data_np[:4]
                
                # 构建标签文本：类别名 + 置信度百分比
                if show_confidence:
                    label = f'{classes[cls_id]} {conf * 100:.0f}%'
                else:
                    label = f'{classes[cls_id]}'
                
                # 绘制水平框
                draw_bbox_on_image(
                    image, 
                    boxes, 
                    colors[cls_id % len(colors)], 
                    label, 
                    font_size_alpha,
                    thickness
                )
                
        except IndexError as e:
            print(f"警告: 检测结果索引错误，跳过: {e}")
            continue
        except Exception as e:
            print(f"警告: 绘制检测框失败: {e}")
            continue