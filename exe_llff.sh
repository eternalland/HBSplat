GPU_ids=0

scenes=(fern flower fortress horns leaves orchids room trex)

neighbor_dis_list=(3 2 3 2 3 3 3 3)
base_thresh_list=("0.1" "0.1" "0.17" "0.1" "0.1" "0.1" "0.1" "0.17")
data_path=/home/mayu/thesis/SCGaussian/data/nerf_llff_data
data_set=LLFF
res=8
output_path=./output/ll_110

input_views=3
run_module=6

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
    match_source=$match_head/$data_set$scene$match_method


    echo ========================= $data_set Train: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene -r $res -m $output_path/$scene \
     --run_module $run_module \
     --input_views $input_views  --neighbor_dis $neighbor_dis \
     --tau_reproj 0.1 --base_thresh $base_thresh --range_sensitivity $base_thresh \
     --eval --switch_generate_matching_mono $switch_generate_matching_mono

    echo ========================= $data_set Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -m $output_path/$scene \
    --input_views $input_views  \

    echo ========================= $data_set Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene

    echo ========================= $data_set Finish: $scene =========================
done

echo ========================= $data_set Average =========================
CUDA_VISIBLE_DEVICES=$GPU_ids python calculate_index.py --base_dir $output_path


echo "experiment end time: $(date)"