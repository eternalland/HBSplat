import json
import os
from collections import defaultdict
import math
import numpy as np
import sys
from argparse import ArgumentParser


def calculate_avg(ssim, psnr, lpips):
    """
    根据公式计算avg值：
    avg = exp(mean(log([10^(-psnr/10), sqrt(1 - ssim), lpips])))
    """
    try:
        # 计算三个项
        psnr_term = 10 ** (-psnr / 10)  # 10^(-psnr/10)
        ssim_term = math.sqrt(1 - ssim)  # sqrt(1 - ssim)
        lpips_term = lpips  # lpips值

        # 创建数组
        terms = [psnr_term, ssim_term, lpips_term]

        # 计算log然后mean然后exp
        log_terms = [math.log(term) for term in terms]
        mean_log = np.mean(log_terms) if len(log_terms) > 0 else 0
        avg_value = math.exp(mean_log)

        return avg_value

    except Exception as e:
        print(f"计算avg值时出错: {e}")
        return 0.0

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--model_path", type=str, default = None)
parser.add_argument("--dataset", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
model_path = args.model_path
dataset = args.dataset
print('model_path: ', model_path)
print('dataset: ', dataset)

# 定义要处理的scene文件夹列表
scenes = []
if "ll" in dataset.lower():
    scenes = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
elif "ib" in model_path.lower():
    scenes = ["zc12", "zc14", "zc15", "zc16", "zc17", "zc18"]
elif "bl" in model_path.lower():
    scenes = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]
elif "dt" in model_path.lower():
    scenes = ["scan8", "scan21", "scan30", "scan31", "scan34", "scan38", "scan40", "scan41", "scan45", "scan55", "scan63", "scan82", "scan103", "scan110", "scan114"]
elif "tt" in model_path.lower():
    scenes = ["Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train"]


head = "ours_2000"
target = 'results'


# 存储所有场景的指标数据
all_metrics = []
metrics_sum = defaultdict(float)
metrics_count = defaultdict(int)

# 创建输出文件路径
output_file = os.path.join(model_path, "metrics_summary.txt")

# 定义列宽（根据实际数据调整）
scene_width = 25  # 增加第一列宽度
metric_width = 12  # 指标列宽度

# 同时输出到控制台和文件
with open(output_file, 'w', encoding='utf-8') as f:
    # 写入表头
    # 写入表头
    header = f"{'Scene':<{scene_width}} {'SSIM':<{metric_width}} {'PSNR':<{metric_width}} {'LPIPS':<{metric_width}} {'AVG':<{metric_width}}\n"
    separator = "-" * (scene_width + 4 * metric_width) + "\n"

    print(header, end='')
    print(separator, end='')
    f.write(header)
    f.write(separator)

    # 遍历每个scene文件夹
    for scene in scenes:
        scene_dir = os.path.join(model_path, scene)
        json_path = os.path.join(scene_dir, f"{target}.json")

        try:
            # 读取JSON文件
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            # 获取指标数据
            # metrics = data.get("ours_2000", {})
            metrics = data.get(head, {})

            if len(metrics) == 0:
                metrics = data.get("ours_10000", {})
            if len(metrics) == 0:
                metrics = data.get("ours_20000", {})
            if len(metrics) == 0:
                metrics = data.get("ours_2500", {})
            if len(metrics) == 0:
                metrics = data.get("ours_4000", {})
            if len(metrics) == 0:
                metrics = data.get("ours_500", {})
            if len(metrics) == 0:
                print("出错")
                sys.exit()
            ssim = metrics.get("SSIM", 0)
            psnr = metrics.get("PSNR", 0)
            lpips = metrics.get("LPIPS", 0)
            avg_val = metrics.get("AVG", 0)

            # 如果JSON中没有AVG值，则手动计算
            if avg_val == 0:
                avg_val = calculate_avg(ssim, psnr, lpips)
                # print(f"为 {scene} 手动计算AVG值: {avg_val:.6f}")


            # 存储当前场景的指标
            scene_metrics = {
                "scene": scene,
                "SSIM": ssim,
                "PSNR": psnr,
                "LPIPS": lpips,
                "AVG": avg_val
            }
            all_metrics.append(scene_metrics)

            # 累加指标值用于计算平均值
            metrics_sum["SSIM"] += ssim
            metrics_sum["PSNR"] += psnr
            metrics_sum["LPIPS"] += lpips

            metrics_count["SSIM"] += 1
            metrics_count["PSNR"] += 1
            metrics_count["LPIPS"] += 1
            metrics_count["AVG"] += 1

            # 格式化输出行
            # 格式化输出行（左对齐）
            output_line = f"{scene:<{scene_width}} {ssim:<{metric_width}.6f} {psnr:<{metric_width}.6f} {lpips:<{metric_width}.6f} {avg_val:<{metric_width}.6f}\n"

            # 输出到控制台和文件
            print(output_line, end='')
            f.write(output_line)

        except FileNotFoundError:
            error_msg = f"{scene:<15} JSON文件未找到: {json_path}\n"
            print(error_msg, end='')
            f.write(error_msg)
        except json.JSONDecodeError:
            error_msg = f"{scene:<15} JSON文件格式错误: {json_path}\n"
            print(error_msg, end='')
            f.write(error_msg)
        except Exception as e:
            error_msg = f"{scene:<15} 读取错误: {str(e)}\n"
            print(error_msg, end='')
            f.write(error_msg)

    # 写入分隔线
    print(separator, end='')
    f.write(separator)

    # 计算并输出平均值
    if all_metrics:
        avg_ssim = metrics_sum["SSIM"] / metrics_count["SSIM"]
        avg_psnr = metrics_sum["PSNR"] / metrics_count["PSNR"]
        avg_lpips = metrics_sum["LPIPS"] / metrics_count["LPIPS"]
        avg_avg = calculate_avg(avg_ssim, avg_psnr, avg_lpips)

        avg_line = f"{'Average':<{scene_width}} {avg_ssim:<{metric_width}.6f} {avg_psnr:<{metric_width}.6f} {avg_lpips:<{metric_width}.6f} {avg_avg:<{metric_width}.6f}\n"

        print(avg_line, end='')
        f.write(avg_line)
    else:
        no_data_msg = "未找到任何有效的指标数据\n"
        print(no_data_msg, end='')
        f.write(no_data_msg)

print(f"\n结果已保存到: {output_file}")