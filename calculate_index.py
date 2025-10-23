import json
import os
from collections import defaultdict
import math
import numpy as np
import sys
from argparse import ArgumentParser


def calculate_avg(ssim, psnr, lpips):
    """
    Calculate avg value based on formula:
    avg = exp(mean(log([10^(-psnr/10), sqrt(1 - ssim), lpips])))
    """
    try:
        # Calculate three terms
        psnr_term = 10 ** (-psnr / 10)  # 10^(-psnr/10)
        ssim_term = math.sqrt(1 - ssim)  # sqrt(1 - ssim)
        lpips_term = lpips  # lpips value

        # Create array
        terms = [psnr_term, ssim_term, lpips_term]

        # Calculate log then mean then exp
        log_terms = [math.log(term) for term in terms]
        mean_log = np.mean(log_terms) if len(log_terms) > 0 else 0
        avg_value = math.exp(mean_log)

        return avg_value

    except Exception as e:
        print(f"Error calculating avg value: {e}")
        return 0.0

parser = ArgumentParser(description="Training script parameters")
parser.add_argument("--model_path", type=str, default = None)
parser.add_argument("--dataset", type=str, default = None)
args = parser.parse_args(sys.argv[1:])
model_path = args.model_path
dataset = args.dataset
print('model_path: ', model_path)
print('dataset: ', dataset)

# Define list of scene folders to process
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


# Store metrics data for all scenes
all_metrics = []
metrics_sum = defaultdict(float)
metrics_count = defaultdict(int)

# Create output file path
output_file = os.path.join(model_path, "metrics_summary.txt")

# Define column widths (adjust based on actual data)
scene_width = 25  # Increase first column width
metric_width = 12  # Metrics column width

# Output to both console and file
with open(output_file, 'w', encoding='utf-8') as f:
    # Write header
    # Write header
    header = f"{'Scene':<{scene_width}} {'SSIM':<{metric_width}} {'PSNR':<{metric_width}} {'LPIPS':<{metric_width}} {'AVG':<{metric_width}}\n"
    separator = "-" * (scene_width + 4 * metric_width) + "\n"

    print(header, end='')
    print(separator, end='')
    f.write(header)
    f.write(separator)

    # Iterate through each scene folder
    for scene in scenes:
        scene_dir = os.path.join(model_path, scene)
        json_path = os.path.join(scene_dir, f"{target}.json")

        try:
            # Read JSON file
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)

            # Get metrics data
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
                print("Error")
                sys.exit()
            ssim = metrics.get("SSIM", 0)
            psnr = metrics.get("PSNR", 0)
            lpips = metrics.get("LPIPS", 0)
            avg_val = metrics.get("AVG", 0)

            # If no AVG value in JSON, manually calculate
            if avg_val == 0:
                avg_val = calculate_avg(ssim, psnr, lpips)
                # print(f"Manually calculated AVG value for {scene}: {avg_val:.6f}")


            # Store current scene metrics
            scene_metrics = {
                "scene": scene,
                "SSIM": ssim,
                "PSNR": psnr,
                "LPIPS": lpips,
                "AVG": avg_val
            }
            all_metrics.append(scene_metrics)

            # Accumulate metric values for calculating average
            metrics_sum["SSIM"] += ssim
            metrics_sum["PSNR"] += psnr
            metrics_sum["LPIPS"] += lpips

            metrics_count["SSIM"] += 1
            metrics_count["PSNR"] += 1
            metrics_count["LPIPS"] += 1
            metrics_count["AVG"] += 1

            # Format output line
            # Format output line (left-aligned)
            output_line = f"{scene:<{scene_width}} {ssim:<{metric_width}.6f} {psnr:<{metric_width}.6f} {lpips:<{metric_width}.6f} {avg_val:<{metric_width}.6f}\n"

            # Output to console and file
            print(output_line, end='')
            f.write(output_line)

        except FileNotFoundError:
            error_msg = f"{scene:<15} JSON file not found: {json_path}\n"
            print(error_msg, end='')
            f.write(error_msg)
        except json.JSONDecodeError:
            error_msg = f"{scene:<15} JSON file format error: {json_path}\n"
            print(error_msg, end='')
            f.write(error_msg)
        except Exception as e:
            error_msg = f"{scene:<15} Read error: {str(e)}\n"
            print(error_msg, end='')
            f.write(error_msg)

    # Write separator line
    print(separator, end='')
    f.write(separator)

    # Calculate and output average values
    if all_metrics:
        avg_ssim = metrics_sum["SSIM"] / metrics_count["SSIM"]
        avg_psnr = metrics_sum["PSNR"] / metrics_count["PSNR"]
        avg_lpips = metrics_sum["LPIPS"] / metrics_count["LPIPS"]
        avg_avg = calculate_avg(avg_ssim, avg_psnr, avg_lpips)

        avg_line = f"{'Average':<{scene_width}} {avg_ssim:<{metric_width}.6f} {avg_psnr:<{metric_width}.6f} {avg_lpips:<{metric_width}.6f} {avg_avg:<{metric_width}.6f}\n"

        print(avg_line, end='')
        f.write(avg_line)
    else:
        no_data_msg = "No valid metrics data found\n"
        print(no_data_msg, end='')
        f.write(no_data_msg)

print(f"\nResults saved to: {output_file}")