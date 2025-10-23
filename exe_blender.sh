GPU_ids=0

dataset=blender
scenes=(chair drums ficus hotdog lego materials mic ship)

data_path=./data/nerf_synthetic
output_path=./output/bl_110
res=2
run_module=6

inv_scale=2
input_views=8
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
     --eval --white_background \
     --run_module $run_module \
     --input_views $input_views  --neighbor_dis 3 --secondary_filtering_number 100 \
     --tau_reproj 0.1 --base_thresh 0.002 --range_sensitivity 0.002  \
     --interpolate_middle_pose_num 2 --virtual_source_num 1 \
     --switch_generate_matching_mono $switch_generate_matching_mono

    if [ $switch_generate_matching_mono -eq 1 ]; then
        continue
    fi

    echo ========================= $dataset Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -m $output_path/$scene$post_fix \
    --input_views $input_views \
    --white_background


    echo ========================= $dataset Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene$post_fix \


    echo ========================= $dataset Finish: $scene =========================
done

echo ========================= $dataset Average =========================
CUDA_VISIBLE_DEVICES=$GPU_ids python calculate_index.py --model_path $output_path --dataset $dataset

echo "experiment end time: $(date)"