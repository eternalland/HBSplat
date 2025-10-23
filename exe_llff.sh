GPU_ids=0

dataset=LLFF
scenes=(fern flower fortress horns leaves orchids room trex)

data_path=/home/mayu/thesis/HBSplat/data/nerf_llff_data
output_path=/home/mayu/thesis/HBSplat/output/ll
res=8
run_module=6

inv_scale=4
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
    neighbor_dis=${neighbor_dis_list[$i]}
    base_thresh=${base_thresh_list[$i]}


    echo ========================= $dataset Train: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene \
     --run_module $run_module \
     --input_views $input_views  --neighbor_dis 5 --secondary_filtering_number 50 \
     --tau_reproj 0.2 --base_thresh 0.17 --range_sensitivity 0.2 \
     --eval \
     --inv_scale $inv_scale --switch_generate_matching_mono $switch_generate_matching_mono

    if [ $switch_generate_matching_mono -eq 1 ]; then
        continue
    fi

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