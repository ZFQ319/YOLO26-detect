# Grad-CAM相关库
import os

import cv2
import numpy as np
import torch
from pytorch_grad_cam import (GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM,
                              HiResCAM, LayerCAM, RandomCAM, EigenGradCAM)
from my_others_code.Grad_CAM import YOLOv8_heatmap
from my_others_code.My_Plot import draw_ground_truth_on_images

# 设置随机种子
np.random.seed(0)
# 选择设备
Device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def main():
    #########需可视化的图像及其数据集设置{数据集：['图像名陈',........]}####################################################
    imgs_data = {
        "SSDD": ['001111.jpg', '000219.jpg'],
        # "RSDD": ['103_18_37.jpg', '90_1_3.jpg'],
        # "HRSID": ['P0128_1800_2600_600_1400.jpg', 'P0120_4200_5000_9000_9800.jpg']
    }
    # ###########设置{模型名称：[需要可视化特征热力图的网络层]}#############################################################
    model_names = {
        # 'yolov5n-obb': [17],
        # 'yolo12n-obb': [14],
        # 'Mamba-YOLO-T-obb': [14],
        # 'yolov8n-obb': [15],
        # 'yolov8n-obb-SD': [15],
        # 'yolov8n-obb-SGAM': [13],
        # 'yolov8n-obb-StrC2f': [15],
        # 'yolov8n-obb-StrC2f-SD': [15],
        # 'yolov8n-obb-SGAM-SD': [13],
        'yolov8n-obb-SGAM-StrC2f-SD': [13]
    }

    # ########## 特征可视化热力图绘制参数设置 ###########################################################################
    task = 'obb'  # 切换为 'bbox' 或 'obb'
    # Grad-CAM方法选择：HiResCAM, LayerCAM, EigenGradCAM, EigenCAM, GradCAM, XGradCAM,, GradCAMPlusPlus RandomCAM等
    method = 'LayerCAM'
    backward_type = 'box'  # 反向目标：class/box/all 或 'range'
    conf_threshold = 0.25  # 置信度阈值
    iou_threshold = 0.5  # IoU阈值，用于NMS
    ratio = 0.03  # 参与计算的Top-K比例（0.02-0.1）
    show_box = True  # 是否绘制检测框
    show_cls = False  # 是否显示类别
    show_conf = True  # 是否显示置信度
    normalize = False  # 是否在框内归一化CAM

    dataset_got_ground_truth = [] # 保存所有图像已经绘制过真实标注的数据，用于后面避免多次绘制
    for dataset in imgs_data:  # 迭代取数据集名
        for model_name in model_names:  # 迭代取网络模型名

            # #############模型权重路径拼接 ############################################################################
            model_path = (f'runs/{task if task == 'obb' else 'detect'}/{dataset}/'
                          f'{model_name}_{dataset}{'-obb' if ((task == 'obb') & (dataset != 'RSDD')) else ''}'
                          f'_640_MaxEpoch300_SGD_lr0.01_seed42/weights/best.pt')
            if not os.path.exists(model_path):
                print(f'未发现【{dataset}】数据集上训练的模型【{model_name}】的路径：【 {model_path}】 ')
                model_path = input(f'请检查并输入【{dataset}】数据集上模型【{model_name}】的正确路径:') + '/weights/best.pt'
            else:
                print(f'发现【{dataset}】数据集上训练的模型【{model_name}】的路径：【 {model_path}】')


            # ###############初始化特在可视化与目标检测预测类 ##############################################################
            feature_cam = YOLOv8_heatmap(
                # 模型权重路径（替换为你的YOLOv8 OBB/bbox模型）
                weight_path=model_path,
                task=task,  # 切换为 'bbox' 或 'obb'
                device=Device,  # 设备选择：'cuda:0' 或 'cpu'
                method=method,  # Grad-CAM方法：GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM等
                layer=model_names[model_name],  # 可视化层索引（根据模型结构调整，如YOLOv8n的第15层）
                backward_type=backward_type,  # 反向目标：class/box/all 或 'range'
                conf_threshold=conf_threshold,  # 置信度阈值
                iou_threshold=iou_threshold,  # IoU阈值
                ratio=ratio,  # 参与计算的Top-K比例（0.02-0.1）
                normalize=normalize,  # 是否在框内归一化CAM
                show_box=show_box,  # 是否绘制检测框
                show_cls=show_cls,
                show_conf=show_conf,
            )

            for img_name in imgs_data[dataset]:  # 迭代取图像名
                # ########################图像、标签路径拼接########################################################
                img_path = (f'../Datasets_detect/{dataset}{'/' + task if dataset != 'RSDD' else ''}'
                            f'/images/all/{img_name}')
                label_path = (f'../Datasets_detect/{dataset}{'/' + task if dataset != 'RSDD' else ''}'
                              f'/labels/test/{img_name[:-4]}.txt')

                # 目标检测结果保存路径设置
                save_path = f'detect_result/{dataset}/'
                save_name = f'{model_name}-{img_name}'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # 读取图像
                original_image = cv2.imread(img_path)

                # ##############################得到带真实标注的图像并保存###################################################
                if dataset not in dataset_got_ground_truth:
                    ground_truth_image = draw_ground_truth_on_images(original_image, label_path)
                    cv2.imwrite(save_path + 'ground-' + img_name[:-4] + '.png', ground_truth_image)

                # ##################################特征可视化热力图绘制并保存##############################################
                original_image: np.ndarray
                heatmap, detect_result__image = feature_cam.get_heatmap(original_image)
                cv2.imwrite(save_path + 'CAM-' + save_name[:-4] + '.png', heatmap)
                cv2.imwrite(save_path + 'DetRes-' + save_name[:-4] + '.png', detect_result__image)

            dataset_got_ground_truth.append(dataset)


if __name__ == '__main__':
    main()
