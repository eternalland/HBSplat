GPU_ids=0

head=/home/mayu/thesis/matched_data/bl_dkm_10000_8_black
scenes=(chair drums ficus hotdog lego materials mic ship)
#scenes=(drums)
#data_set=bl_
data_set=""
#match_method=000_0_10000_8
match_method=


neighbor_dis_list=(3 3 3 3 3 3 3 3)
#neighbor_dis_list=(2 2 2 2 2 2 2 2)
data_path=/home/mayu/thesis/SCGaussian/data/nerf_synthetic
output_path=./oo4/bl_dkm110_0_10000_8_proj0_nei3_forward_110-11
further=
res=2
run_module=6
switch_fast_other_mono_match=1
input_views=8
post_fix=110_0_10000_8


mkdir -p $output_path


for i in "${!scenes[@]}"
do
    scene=${scenes[$i]}
    neighbor_dis=${neighbor_dis_list[$i]}

    log_file_path=$output_path/loss_$scene.txt
    touch "$log_file_path"

    echo ========================= $data_set Train: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python train.py -s $data_path/$scene/$further -r $res -m $output_path/$scene$post_fix \
     --run_module $run_module \
     --input_views $input_views  --neighbor_dis $neighbor_dis \
     --tau_reproj 0.1 --base_thresh 0.002 --range_sensitivity 0.002 --opacity 0.005 \
     --eval --white_background

    echo ========================= $data_set Render: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python render.py -m $output_path/$scene$post_fix \
    --input_views $input_views  \
    --fast_other_mono $head/$data_set$scene$match_method/mono_depth_map \
    --fast_other_match $head/$data_set$scene$match_method/matched_image/matched_data.npy --white_background\


    echo ========================= $data_set Metric: $scene =========================
    CUDA_VISIBLE_DEVICES=$GPU_ids python metrics.py -m $output_path/$scene$post_fix \


    echo ========================= $data_set Finish: $scene =========================
done


