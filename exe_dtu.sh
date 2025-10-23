GPU_ids=0

dataset=dtu
scenes=(scan8 scan21 scan30 scan31 scan34 scan38 scan40 scan41 scan45 scan55 scan63 scan82 scan103 scan110 scan114)

data_path=./data/dtu
output_path=./output/dtu_110
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

    python train.py -s $data_path/$scene/$further -r $res -m $output_path/$scene \
    --eval \
    --run_module $run_module \
    --input_views $input_views  --neighbor_dis 3 --inv_scale $inv_scale \
    --tau_reproj 0.05 --base_thresh 0.05 --range_sensitivity 0.1 \
    --switch_generate_matching_mono $switch_generate_matching_mono \

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

