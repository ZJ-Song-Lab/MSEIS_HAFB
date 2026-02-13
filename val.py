import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    # 使用改进后的模型权重进行验证
    model_path = 'runs/train/mseis_hafb_exp/weights/best.pt'
    model = RTDETR(model_path)
    
    # 验证改进后的模型
    result = model.val(data='data.yaml',  # 数据集配置文件路径
                      split='test',
                      imgsz=640,
                      batch=4,
                      project='runs/val',
                      name='mseis_hafb_val',  # 验证实验名称
                      )
    
    if model.task == 'detect':
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        
        n_l, n_p, n_g, flops = model_info(model.model)
        
        model_info_table = PrettyTable()
        model_info_table.title = "Model Info (MSEIS-HAFB)"
        model_info_table.field_names = ["GFLOPs", "Params", "Preprocess", "Inference", "Postprocess", "FPS", "Inference FPS", "Model Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}', 
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s', 
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}', 
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrics (MSEIS-HAFB)"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row([
                                        cls_name, 
                                        f"{result.box.p[idx]:.4f}", 
                                        f"{result.box.r[idx]:.4f}", 
                                        f"{result.box.f1[idx]:.4f}", 
                                        f"{result.box.ap50[idx]:.4f}", 
                                        f"{result.box.all_ap[idx, 5]:.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                        f"{result.box.ap[idx]:.4f}"
                                    ])
        # 添加总体指标
        model_metrice_table.add_row([
                                    "Overall",
                                    f"{result.results_dict['metrics/precision(B)']:.4f}",
                                    f"{result.results_dict['metrics/recall(B)']:.4f}", 
                                    f"{np.mean(result.box.f1):.4f}", 
                                    f"{result.results_dict['metrics/mAP50(B)']:.4f}", 
                                    f"{np.mean(result.box.all_ap[:, 5]):.4f}", # 50 55 60 65 70 75 80 85 90 95 
                                    f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
                                ])
        print(model_metrice_table)

        # 保存验证结果到文件
        with open(result.save_dir / 'mseis_hafb_validation_results.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
        
        print(f"Validation results saved to: {result.save_dir / 'mseis_hafb_validation_results.txt'}")
        
