import math
import time

import cv2
import numpy as np
import torch
# Grad-CAM相关库
from pytorch_grad_cam import (LayerCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from tqdm import tqdm

from my_others_code.My_Plot import draw_obb_on_images, draw_bbox_on_image
# YOLOv8相关库
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xywh2xyxy, xywhr2xyxyxyxy


def main():
    # 设置随机种子
    np.random.seed(0)
    # 选择设备
    Device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = ''
    img_path = ''

    feature_cam = YOLOv8_heatmap(
        # 模型权重路径（替换为你的YOLOv8 OBB/bbox模型）
        weight_path=model_path,
        task='obb',  # 切换为 'bbox' 或 'obb'
        device=Device,  # 设备选择：'cuda:0' 或 'cpu'
        method=LayerCAM,  # Grad-CAM方法：GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM等
        layer=15,  # 可视化层索引（根据模型结构调整，如YOLOv8n的第15层）
        backward_type='box',  # 反向目标：class/box/all 或 'range'
        conf_threshold=0.25,  # 置信度阈值
        iou_threshold=0.5,  # IoU阈值
        ratio=0.05,  # 参与计算的Top-K比例（0.02-0.1）
        normalize=False,  # 是否在框内归一化CAM
        show_box=True,  # 是否绘制检测框
        show_cls=False,
        show_conf=True,
    )


    original_image = cv2.imread(img_path)
    # ##################################特征可视化热力图绘制并保存##############################################
    original_image: np.ndarray
    feature_cam.get_heatmap(original_image)



def get_ground_truth(image, label_path):
    """在原始图像上绘制标签"""
    original_image = np.array(image).copy()
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
            thickness = 2
            cv2.line(original_image, point[0], point[1], color, thickness)
            cv2.line(original_image, point[1], point[2], color, thickness)
            cv2.line(original_image, point[2], point[3], color, thickness)
            cv2.line(original_image, point[3], point[0], color, thickness)
    return original_image


class ActivationsAndGradients:
    def __init__(self, model_explained, task, target_layers, reshape_transform):
        """
        用于提取指定中间层的激活特征和梯度信息（保留梯度传播链）
       Args：
            model_explained: 被解释的模型
            task: 'bbox' or 'obb'
            target_layers: 可视化特征之前的网络层
            reshape_transform:对激活和梯度进行 reshape 的函数（适配检测模型）
        """
        self.model = model_explained  # 被解释的模型
        self.gradients = []  # 保存梯度（CPU）
        self.activations = []  # 保存激活特征（CPU）
        self.reshape_transform = reshape_transform  # 对激活和梯度进行 reshape 的函数
        self.handles = []  # hook 句柄，用于释放
        self.task = task
        for target_layer in target_layers:  # 循环注册hook
            # 保存激活特征
            self.handles.append(target_layer.register_forward_hook(self.save_activation))
            # 保存梯度
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def __call__(self, x):
        """执行前向并返回目标（保留梯度传播，处理列表输出）"""
        self.gradients = []  # 清空历史梯度
        self.activations = []  # 清空历史激活
        # # 临时切换到训练模式（保留梯度，BN层不影响CAM）
        model_output = self.model(x)[0]

        # 单张量情况：去掉批次维度
        result = model_output.permute(0, 2, 1).squeeze(0)

        # 对 YOLO 输出进行后处理（保留梯度）
        post_result, pre_post_boxes, post_boxes = self.post_process(result=result, task=self.task)

        # 返回用于 CAM 计算的结果（原设备，带梯度）
        return [[post_result, pre_post_boxes]]

    def save_activation(self, _, __, output):
        """保存前向激活（存储到CPU，原张 量保留梯度）"""
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        # 存储时detach到CPU，不影响原计算图
        self.activations.append(activation.detach().cpu())

    def save_gradient(self, _, __, output):
        """只有 requires_grad=True 的 Tensor 才能注册 hook"""
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        # 定义梯度保存函数
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            # 梯度存储到CPU，原梯度保留在计算图中
            self.gradients = [grad.detach().cpu()] + self.gradients

        # 注册梯度hook
        output.register_hook(_store_grad)

    @staticmethod
    def post_process(result, task='bbox'):
        """YOLO 输出后处理（核心）- 修复批次维度+保留梯度"""
        # 处理空张量
        if len(result) == 0:
            return result, result, torch.tensor([])

        if task == 'obb':
            idx = [0, 1, 2, 3, 5, 4]
            result = result[:, idx]
            logits_ = result[:, 5:]  # 分类得分 (num_anchors, num_classes)
            boxes_ = result[:, :5]  # 边界框 (x、y、w、h、r) (num_anchors, 5)
        else:
            logits_ = result[:, 4:]  # 分类得分 (num_anchors, num_classes)
            boxes_ = result[:, :4]  # 边界框 (x、y、w、h) (num_anchors, 4)

        # 数值校验：替换异常值（不破坏计算图）
        boxes_ = torch.clamp(boxes_, min=0.0)  # 负数替换为0
        boxes_ = torch.where(torch.isinf(boxes_), torch.zeros_like(boxes_), boxes_)
        boxes_ = torch.where(torch.isnan(boxes_), torch.zeros_like(boxes_), boxes_)

        # 处理空数据情况
        if len(logits_) == 0 or len(boxes_) == 0:
            return logits_, boxes_, torch.tensor([])

        # 按最大类别置信度排序（保留梯度）
        max_conf, indices = torch.sort(logits_.max(1)[0], descending=True)
        logits_sorted = logits_[indices]  # (num_anchors, num_classes)
        boxes_sorted = boxes_[indices]  # (num_anchors, 5/4)

        # 转换为像素坐标（仅用于返回，脱离计算图）
        if len(boxes_sorted) == 0:
            box_xyi = torch.tensor([])
        else:
            if task == 'obb':
                box_xyi = xywhr2xyxyxyxy(boxes_sorted).detach()  # (num_anchors, 8)
            else:
                box_xyi = xywh2xyxy(boxes_sorted).detach()  # (num_anchors, 4)

        return logits_sorted, boxes_sorted, box_xyi

    def release(self):
        """释放 hook（防止显存泄漏）"""
        for handle in self.handles:
            handle.remove()


class YOLOv8_target(torch.nn.Module):

    def __init__(self, output_type, conf, ratio) -> None:
        """
        Grad-CAM 目标函数构造模块，用于从 YOLOv8 的检测输出中，根据置信度阈值与指定输出类型（类别分数 / 边界框 / 二者）构造一个 可反向传播的标量目标值。
        该目标值将作为 Grad-CAM 的反向传播起点，从而引导梯度集中于高置信度目标区域，实现对检测模型决策区域的可解释性可视化。
        Args:
            output_type: Grad-CAM 目标类型('class'、'box' 或 'all')
            conf: 置信度阈值，低于该值的检测结果将被忽略
            ratio: 参与 Grad-CAM计算的目标比例（Top-K）
        """
        super().__init__()
        self.output_type = output_type  # Grad-CAM 目标类型：class / box / all
        self.conf = conf  # 置信度阈值
        self.ratio = ratio  # 使用前 ratio 比例的高置信度目标

    def forward(self, data):
        """
        data: [post_result, pre_post_boxes]
        post_result: (N, num_classes) 类别得分
        pre_post_boxes: 【(N, 4) HBB边界框 (x、y、w、h 或 xy、xy)】 或【(N, 5) OBB边界框 (x、y、w、h、range)】
        """
        post_result, pre_post_boxes = data
        result = []  # 存储用于反向传播的标量分量
        time.sleep(0.5)
        # 只对前 ratio 比例的高置信度目标计算 CAM
        for i in tqdm(range(int(post_result.size(0) * self.ratio)), 
               desc='特征梯度映射ing', 
               bar_format='{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {elapsed}<{remaining}'):

            # 若当前目标的最大类别置信度低于阈值，则终止
            if float(post_result[i].max()) < self.conf:
                break

            # 以类别置信度作为 Grad-CAM 目标
            if self.output_type == 'class' or self.output_type == 'all':
                result.append(post_result[i].max())

            # 以边界框回归值作为 Grad-CAM 目标
            elif self.output_type == 'box' or self.output_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.output_type == 'range' or self.output_type == 'all':
                result.append(pre_post_boxes[i, 4])

        # 将所有目标值求和，构成标量用于反向传播
        return sum(result)


class YOLOv8_heatmap:
    """ YOLOv8 OBB/bbox Grad-CAM 可视化工具类"""
    def __init__(self, weight_path, task, device, method, layer, backward_type,
                 conf_threshold=0.25, iou_threshold=0.5, ratio=0.02,
                 normalize=False, show_box=False, show_cls=False, show_conf=False):
        """类初始化
        Args:
            weight_path: 模型权重路径
            task: 检测任务类型 'bbox' or 'obb'
            device: 推理设备（'cuda:0' / 'cpu'）
            method: Grad-CAM方法名（如 'GradCAM'、'EigenCAM'）
            layer: 可视化特征所在网络层索引列表
            backward_type: 反向目标类型（'class'、'box'、'all'）
            conf_threshold: 检测置信度阈值
            iou_threshold: 检测IoU阈值用于NMS
            ratio: 参与 CAM 计算的目标比例
            normalize: 是否在框内重新归一化 CAM
            show_box: 是否绘制检测框
            show_cls: 是否显示类别
            show_conf: 是否显示置信度
        """
        self.show_cls = show_cls
        self.show_conf = show_conf
        self.show_box = show_box
        self.conf_threshold = torch.tensor(conf_threshold, requires_grad=True)  # 转为张量
        self.iou_threshold = iou_threshold
        self.normalize = normalize
        self.task = task  # 'bbox' or 'obb'
        self.device = torch.device(device)

        # 加载YOLOv8模型（保留梯度）, 读取类别名称
        self.model, ckpt = load_checkpoint(weight_path, device)
        self.model_names = ckpt['model'].names
        self.model.info()  # 打印模型信息
        # 强制开启所有参数的梯度（包括冻结层）
        for p in self.model.parameters():
            p.requires_grad_(True)
            p.retain_grad()  # 强制保留梯度
        self.model.eval()  # 评估模式（后续在forward中临时切换train）

        # 构造Grad-CAM目标函数
        self.target = YOLOv8_target(backward_type, self.conf_threshold, ratio)
        # 选择目标层
        target_layers = [self.model.model[i] for i in layer]
        # 初始化Grad-CAM方法
        self.method = eval(method)(self.model, target_layers)

        # 绑定激活与梯度提取器
        self.method.activations_and_grads = ActivationsAndGradients(
            self.model, self.task, target_layers, None
        )

        # 为不同类别生成颜色
        rng = np.random.default_rng(42)

        self.colors = []

        for i in range(len(self.model_names)):
            if i == 0:
                self.colors.append((255, 0, 0))  # 红
            elif i == 1:
                self.colors.append((0, 255, 0))  # 绿
            elif i == 2:
                self.colors.append((0, 0, 255))  # 蓝
            else:
                color = tuple(rng.integers(80, 255, 3).tolist())
                self.colors.append(color)

    def get_heatmap(self, original_image):
        """单张图像 CAM 生成"""
        # 读取图像
        img = original_image.copy()
        # Letterbox预处理
        img, ratio, (dw, dh) = self.letterbox(img)

        # 图像格式转换：BGR → RGB → 归一化
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img_rgb) / 255.0

        # 转换为张 量（开启梯度）
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        tensor.requires_grad_(True)  # 开启输入张量梯度
        tensor.retain_grad()  # 保留输入梯度

        # 计算Grad-CAM热力图
        grayscale_cam = self.method(tensor, [self.target])

        # 提取CAM并叠加到图像
        grayscale_cam = grayscale_cam[0, :]
        torch.cuda.empty_cache()
        cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)

        # YOLO推理获取检测结果（修复列表输出）
        self.model.eval()
        with torch.no_grad():
            result = self.model(tensor)
        result = result[0]
        pre = self.post_process(result)

        if self.task == 'obb':
            idx = [0, 1, 2, 3, 6, 4, 5]
            pre = pre[:, idx]

        # 框内CAM归一化
        if self.normalize and pre is not None and len(pre) > 0:
            if self.task == 'obb':
                boxes = xywhr2xyxyxyxy(pre[:, :5]).detach().cpu().numpy().astype(np.int32)
                cam_image = self.normalize_cam_in_obb(boxes, img_float, grayscale_cam)
            else:
                boxes = pre[:, :4].detach().cpu().numpy().astype(int)
                cam_image = self.normalize_cam_in_bounding_boxes(boxes, img_float, grayscale_cam)
        h0, w0 = original_image.shape[:2]

        # 在特在可视化热力图上绘制检测框、类别、置信度
        if pre is not None and len(pre) > 0:
            for data in pre:
                data_np = data.detach().cpu().numpy()
                # 提取类别和置信度
                if self.task == 'obb':
                    cls_id = int(data_np[6])
                    conf = float(data_np[5])
                    boxes = xywhr2xyxyxyxy(data_np[:5])
                else:
                    cls_id = int(data_np[4:].argmax())
                    conf = float(data_np[4:].max())
                    boxes = data_np[:4]

                if self.show_cls & self.show_conf:
                    label = f'{self.model_names[cls_id]} {conf * 100:.0f}'
                elif self.show_cls:
                    label = f'{self.model_names[cls_id]}'
                elif self.show_conf:
                    label = f'{conf * 100:.0f}'
                else:
                    label = ''
                # 绘制框
                img: np.ndarray
                if self.task == 'obb':
                    if self.show_box:
                        cam_image = draw_obb_on_images(cam_image, boxes, self.colors[cls_id], label, 5)
                    img = draw_obb_on_images(img, boxes, self.colors[cls_id], label, 5)
                else:
                    if self.show_box:
                        cam_image = draw_bbox_on_image(cam_image, boxes, self.colors[cls_id], label, 5)
                    img = draw_bbox_on_image(img, boxes, self.colors[cls_id], label, 5)

        # 将图像去除填充部分，还原到原始尺寸和RGB格式################################################################
        # ===== 精确计算padding =====
        top = int(round(dh + 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw + 0.1))
        right = int(round(dw + 30))
        # ===== 去掉padding =====
        h, w = cam_image.shape[:2]
        cam_image = cam_image[top:h - bottom, left:w - right]
        detect_result_image = img[top:h - bottom, left:w - right]
        # ===== resize回原图 =====
        cam_image = cv2.resize(cam_image, (w0, h0))
        detect_result_image = cv2.resize(detect_result_image, (w0, h0))
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        detect_result_image = cv2.cvtColor(detect_result_image, cv2.COLOR_RGB2BGR)

        return cam_image, detect_result_image

    def post_process(self, result):
        """YOLO输出后处理（NMS，适配OBB/bbox，处理列表输出）"""
        # 处理模型输出为列表的情况
        if isinstance(result, list):
            result = result[0]

        if self.task == 'obb':
            # OBB的NMS处理
            pre = non_max_suppression(
                result,
                nc=1,
                conf_thres=self.conf_threshold.item(),  # 转为标量
                iou_thres=self.iou_threshold,
                rotated=True
            )[0]
        else:
            # BBOX的NMS处理
            pre = non_max_suppression(
                result,
                conf_thres=self.conf_threshold.item(),
                iou_thres=self.iou_threshold
            )[0]

        # 处理空结果和批次维度
        if pre is None:
            return torch.tensor([])
        pre = pre.squeeze(0) if len(pre.shape) > 2 else pre
        return pre


    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(0, 0, 0),
                  auto=True, scale_fill=False, scaleup=True, stride=32):
        """
        在满足跨步多重约束的同时调整图像大小和填充图像，确保图像不变形的情况下实现缩放，达到模型输入尺寸要求
        Args:
            img: cv2.UMat | np.ndarray 图像
            new_shape:=(w, h)调整后的尺度
            color:=(R, G, B):短边填充颜色
            auto:=True:是否自动对齐 stride（True：最终图像尺寸 一定是 stride 的整数倍 ，False：严格等于 new_shape）
            scale_fill:=False:是否强制拉伸填满（True：图像被拉伸，宽高比改变 ；False：保持比例（推荐））
            scaleup:=False:是否允许放大图像（False：小图只 padding，不放大有利于提升验证 mAP，True：小图也会被放大）
            stride:=32:网络最大下采样步长
        """
        shape = img.shape[:2]  # 当前形状 [高度，宽度]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 比例（新旧）
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小，不要放大（为了更好的 val mAP）
            r = min(r, 1.0)

        # 计算填充
        ratio = r, r  # 宽度、高度比例
        new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_pad[0], new_shape[0] - new_pad[1]  # wh 填充
        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh 填充  np.mod(dw, stride) 等价于 dw % stride
        elif scale_fill:  # 延伸
            dw, dh = 0.0, 0.0
            new_pad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

        dw /= 2  # 将填充物分为两边
        dh /= 2

        if shape[::-1] != new_pad:  # 调整尺寸
            img = cv2.resize(img, new_pad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边界
        return img, ratio, (dw, dh)  # 返回尺度调整后的图像大小， 缩放比， 短边填充像素大小


    @staticmethod
    def normalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
        """轴对齐框（bbox）的CAM局部归一化"""
        normalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        cam_h, cam_w = grayscale_cam.shape

        for x1, y1, x2, y2 in boxes:
            # 坐标越界处理
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(cam_w - 1, x2), min(cam_h - 1, y2)
            # 局部归一化
            normalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())

        # 全局归一化
        normalized_cam = scale_cam_image(normalized_cam)
        return show_cam_on_image(image_float_np, normalized_cam, use_rgb=True)

    @staticmethod
    def normalize_cam_in_obb(boxes_8p, image_float_np, grayscale_cam):
        """旋转框（OBB）的CAM局部归一化（多边形区域）"""
        normalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        cam_h, cam_w = grayscale_cam.shape

        for box in boxes_8p:
            # 重塑为4x2的多边形坐标
            poly = box.reshape(4, 2).astype(np.int32)
            # 创建多边形掩码
            mask = np.zeros((cam_h, cam_w), np.uint8)
            cv2.fillPoly(mask, [poly], (1, 1, 1))
            # 提取多边形内的CAM并归一化
            cam_poly = grayscale_cam * mask
            non_zero = cam_poly[mask == 1]
            if len(non_zero) > 0:
                cam_poly[mask == 1] = scale_cam_image(non_zero.reshape(1, -1)).reshape(-1)
                normalized_cam += cam_poly

        # 全局归一化
        normalized_cam = scale_cam_image(normalized_cam)
        return show_cam_on_image(image_float_np, normalized_cam, use_rgb=True)


if __name__ == '__main__':
    main()
