import math
import cv2
import numpy as np

def draw_bbox_on_image(img, box, color, label, label_font_size_alpha=1.0):
    """
    绘制HBB边框和标签信息（水平矩形框）

    Args:

        img: 图像（numpy数组，会被直接修改）
        box: 边界框四个坐标值 [x1, y1, x2, y2]
        color: 框线颜色（BGR格式的列表，如[255,0,0]表示红色）
        label: 标签内容（字符串）
        label_font_size_alpha: 类别标签和置信度的字号调整系数
    Returns:
        img: 绘制后的图像（与输入是同一个对象）
    """

    # 将坐标转换为整数类型（OpenCV的坐标必须是整数）
    x1, y1, x2, y2 = list(map(int, box))
    # 在图像上绘制矩形框
    # 参数：图像、左上角坐标、右下角坐标、颜色、线宽、线型
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=1, lineType=cv2.LINE_AA)
    # 根据框的宽度计算字体大小（相对比例）
    # 计算边框和图像的对角线长
    obb_l = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    img_l = math.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2)
    # 计算边长占图像高度的比例
    a = obb_l / img_l
    # 字体大小计算公式：50 * a^2 + 0.4
    font_size =label_font_size_alpha*a ** 2 + 0.4
    # 绘制类别标签（调用plot_label方法）
    show_label_on_image(img, label, x1, y1, font_size, color)
    return img

def draw_obb_on_images(img, box_8p, color, label, label_font_size_alpha=1.0):
    """
    绘制OBB边框和标签信息（旋转矩形框）

    Args:
        img: 图像（numpy数组，会被直接修改）
        box_8p: OBB框8个顶点坐标（4个点的x,y坐标连续排列）
        color: 框线颜色（BGR格式的列表）
        label: 标签内容（字符串）
        label_font_size_alpha: 类别标签和置信度的字号调整系数
    Returns:
        img: 绘制后的图像（与输入是同一个对象）
    """

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
    return img

def show_label_on_image(img, label,
                        x, y, font_size=1.0, color=None, alpha=0.3):
    """
    绘制标签函数（静态方法）
    功能：绘制带半透明背景的文本标签，文字自适应调整位置

    Args:
        img: 图像（numpy数组，会被直接修改）
        label: 标签内容（字符串）
        x: 标签显示位置的x坐标（左上角）
        y: 标签显示位置的y坐标（顶部）
        font_size: float 字号大小，默认为1
        color: 背景矩形的颜色（BGR），默认红色
        alpha: 背景透明度，范围0-1，默认0.3（30%不透明）

    Returns:
        img: 绘制后的图像
    """
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
    return img

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