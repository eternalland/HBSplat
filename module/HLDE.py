from tqdm import tqdm
import torch
from utils import filter_outlier

def initial_depth_estimation(gaussians, sparse_args, iter_start, iter_end, viewpoint_stack):

    # init stage
    if 0b100 & sparse_args.run_module:
        gaussians.training_setup_init5()
    else:
        gaussians.training_setup_init()

    ema_loss_for_log = 0.0

    ran = range(sparse_args.num_iterations)

    progress_bar = tqdm(ran, desc="Init progress")

    best_state_dict = None
    min_loss_state = None
    propagate_loss = 0.

    for iteration in ran:
        if iteration in [500, 1000, 1500]:
            gaussians.update_learning_rate_init(0.5)

        iter_start.record()

        matchloss, loss_state = gaussians.get_matchloss_from_base()
        loss = 5 * matchloss

        if 0b100 & sparse_args.run_module:
            if iteration == sparse_args.start_propagate_iteration:
                print("\nEnable get_propagate_loss2:", sparse_args.start_propagate_iteration, sparse_args.propagate_weight)
            if iteration >= sparse_args.start_propagate_iteration:
                propagate_loss, loss_state = gaussians.get_propagate_loss3(loss_state)
                loss += sparse_args.propagate_weight * propagate_loss

        if 0b010 & sparse_args.run_module and sparse_args.switch_small_transform and sparse_args.small_transform_iter == iteration:
            gaussians.load_z_val5(best_state_dict)

        if best_state_dict is None:
            best_state_dict = gaussians.get_z_val()
            min_loss_state = loss_state
        else:
            curr_state_dict = gaussians.get_z_val()
            for key, v in loss_state.items():
                for key1, v1 in loss_state[key].items():
                    best_state_dict[key][key1] = torch.where((min_loss_state[key][key1] < loss_state[key][key1]).unsqueeze(-1), best_state_dict[key][key1], curr_state_dict[key][key1])
                    min_loss_state[key][key1] = torch.where(min_loss_state[key][key1] < loss_state[key][key1], min_loss_state[key][key1], loss_state[key][key1])

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # if iteration >= 1999:
            #     progress_bar.close()

            if 0b100 & sparse_args.run_module:
                if iteration % 10 == 0:
                    # ratio = matchloss / smooth_loss
                    progress_bar.set_postfix({
                        "mat_l": f"{matchloss:.4f}",
                        "mat_a": f"{5 * matchloss:.4f}",
                        "pro_l": f"\n{propagate_loss:.4f}",
                        "pro_a": f"{sparse_args.propagate_weight * propagate_loss:.4f}",
                        "Total": f"{loss:.4f}",
                    })

            # Optimizer step
            gaussians.optimizer_init.step()
            gaussians.optimizer_init.zero_grad(set_to_none=True)

        if 0b100 & sparse_args.run_module:

            if iteration == sparse_args.filter_iteration - 1:  # Because iteration starts from 0

                if 'blender' in sparse_args.dataset:
                    filter_outlier.filter_propagate_data_common(gaussians, min_loss_state, sparse_args)
                    if sparse_args.switch_second_filter:
                        filter_outlier.second_filter(gaussians, sparse_args)
                else:
                    progress_bar.close()
                    print(f"\nPerforming filtering at iteration {sparse_args.filter_iteration}")

                    filter_outlier.filter_outliers(gaussians, sparse_args, min_loss_state)
                    if sparse_args.switch_second_filter:
                        filter_outlier.second_filter(gaussians, sparse_args)

                    gaussians.training_setup_init()

                    # Reset state (optional)
                    ema_loss_for_log = 0.0
                    best_state_dict = None
                    min_loss_state = None

                    # Create new progress bar
                    progress_bar = tqdm(range(sparse_args.filter_iteration, sparse_args.num_iterations),
                                        desc="Post-filter training",
                                        initial=sparse_args.filter_iteration)


    gaussians.load_z_val(best_state_dict)

    if 'blender' in sparse_args.dataset:
        filter_outlier.filter_outside(gaussians, viewpoint_stack, min_loss_state, sparse_args)

    return min_loss_state