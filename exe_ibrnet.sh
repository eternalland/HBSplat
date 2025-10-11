GPU_ids=0

dataset=ibrnet
scenes=(zc12 zc14 zc15 zc16 zc17 zc18)

data_path=./data/ibrnet
output_path=./output/ib_110/
res=2
run_module=6

inv_scale=1
input_views=3
switch_generate_matching_mono=0


mkdir -p $output_path
log_file="$output_path/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$log_file") 2>&1
echo "experiment start time: $(date)"
echo "$log_file path: $log_file"
touch "$log_file"



for i in "${!scenes[@]}"
do
    scene=${scenes[$i]}

    echo ========================= $dataset Train: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene \
     --run_module $run_module \
     --input_views $input_views  --neighbor_dis 2 \
     --tau_reproj 0.1 --base_thresh 0.2 --range_sensitivity 0.1 \
     --eval \
     --switch_generate_matching_mono $switch_generate_matching_mono

    echo ========================= $dataset Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -m $output_path/$scene \
    --input_views $input_views  \

    echo ========================= $dataset Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= $dataset Finish: $scene =========================
done

echo ========================= $dataset Average =========================
CUDA_VISIBLE_DEVICES=$GPU_ids python calculate_index.py --model_path $output_path --dataset $dataset

echo "experiment end time: $(date)"