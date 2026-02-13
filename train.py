import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    # 使用改进的模型配置文件（包含MSEIS和HAFB模块）
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-MSEIS-HAFB.yaml')
    # model.load('') # 加载预训练权重
    
    # 训练改进后的模型
    model.train(data='data.yaml',  # 数据集配置文件路径
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4,
                workers=4,
                # device='0,1',  # 多GPU训练
                # resume='',  # 恢复训练
                project='runs/train',
                name='mseis_hafb_exp',  # 实验名称
                )