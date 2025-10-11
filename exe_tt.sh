GPU_ids=0

dataset=tt
scenes=(Family Francis Horse Lighthouse M60 Panther Playground Train)

data_path=./data/colmap_Tanks_Temple
output_path=./output/tt_110
res=4
run_module=4

inv_scale=2
input_views=24
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

    log_file_path=$output_path/loss_$scene.txt
    touch "$log_file_path"

    echo ========================= $dataset Train: $scene =========================

    python train.py -s $data_path/$scene/$further -r $res -m $output_path/$scene \
    --eval \
    --run_module $run_module \
    --input_views $input_views  --neighbor_dis 2 --inv_scale $inv_scale \
    --tau_reproj 0.001 --base_thresh 0.01 --range_sensitivity 0.1 \
    --switch_generate_matching_mono $switch_generate_matching_mono \

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
