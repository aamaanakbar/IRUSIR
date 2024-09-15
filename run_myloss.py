import datetime
import glob
import os
import tqdm
import torch
from torch.optim import Adam
from cli import parse_config
import benchmark
from benchmark import tasks, Degradation

from benchmark import tasks, Degradation
from robust_unsupervised import *
 # Make sure to import the updated loss function here

config = parse_config()
benchmark.config.resolution = config.resolution

print(config.name)
timestamp = datetime.datetime.now().isoformat(timespec="seconds").replace(":", "")

G = open_generator(config.pkl_path)
loss_fn = MultiscaleLPIPS()

def run_phase(label: str, variable: Variable, lr: float):
    # Run optimization loop
    optimizer = Adam(variable.parameters(), lr=lr)  # Changed to Adam optimizer
    try:
        for _ in tqdm.tqdm(range(150), desc=label):
            x = variable.to_image()
            # Make sure `degradation` and `target` are correctly initialized
            loss = loss_fn(degradation.degrade_prediction, x, target, degradation.mask).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    except KeyboardInterrupt:
        pass

    # Log results
    suffix = "_" + label
    pred = resize_for_logging(variable.to_image(), config.resolution)

    approx_degraded_pred = degradation.degrade_prediction(pred)
    degraded_pred = degradation.degrade_ground_truth(pred)

    save_image(pred, f"pred{suffix}.png", padding=0)
    save_image(degraded_pred, f"degraded_pred{suffix}.png", padding=0)

    save_image(
        torch.cat([approx_degraded_pred, degraded_pred]),
        f"degradation_approximation{suffix}.png",
        padding=0,
    )

    save_image(
        torch.cat(
            [
                ground_truth,
                resize_for_logging(target, config.resolution),
                resize_for_logging(degraded_pred, config.resolution),
                pred,
            ]
        ),
        f"side_by_side{suffix}.png",
        padding=0,
    )
    save_image(
        torch.cat([resize_for_logging(target, config.resolution), pred]),
        f"result{suffix}.png",
        padding=0,
    )
    save_image(
        torch.cat([target, degraded_pred, (target - degraded_pred).abs()]),
        f"fidelity{suffix}.png",
        padding=0,
    )
    save_image(
        torch.cat([ground_truth, pred, (ground_truth - pred).abs()]),
        f"accuracy{suffix}.png",
        padding=0,
    )

if __name__ == '__main__':
    if config.tasks == "single":
        tasks = benchmark.single_tasks
    elif config.tasks == "composed":
        tasks = benchmark.composed_tasks
    elif config.tasks == "all":
        tasks = benchmark.all_tasks
    elif config.tasks == "custom":
        # Implement your own degradation here
        class YourDegradation:
            def degrade_ground_truth(self, x):
                "The true degradation you are attempting to invert."
                raise NotImplementedError
            
            def degrade_prediction(self, x):
                """
                Differentiable approximation to the degradation in question. 
                Can be identical to the true degradation if it is invertible.
                """
                raise NotImplementedError
        tasks = [
            benchmark.Task(
                constructor=YourDegradation,
                # These labels are just for the output folder structure
                name="your_degradation", 
                category="single", 
                level="M", 
            )
        ]
    else:
        raise Exception("Invalid task name")
    


    for task in tasks:
        experiment_path = f"out/{config.name}/{timestamp}/{task.category}/{task.name}/{task.level}/"
        
        image_paths = sorted(
            [
                os.path.abspath(path)
                for path in (
                    glob.glob(config.dataset_path + "/**/*.png", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.jpg", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.jpeg", recursive=True)
                    + glob.glob(config.dataset_path + "/**/*.tif", recursive=True)
                )
            ]
        )
        assert len(image_paths) > 0, "No images found!"

        with directory(experiment_path):
            print(experiment_path)
            print(os.path.abspath(config.dataset_path))

            for j, image_path in enumerate(image_paths):
                with directory(f"inversions/{j:04d}"):
                    print(f"- {j:04d}")
                    
                    ground_truth = open_image(image_path, config.resolution)
                    degradation = task.init_degradation()
                    save_image(ground_truth, f"ground_truth.png")
                    target = degradation.degrade_ground_truth(ground_truth)
                    save_image(target, f"target.png")
                    
                    W_variable = WVariable.sample_from(G)
                    run_phase("W", W_variable, config.global_lr_scale * 0.08)

                    Wp_variable = WpVariable.from_W(W_variable)
                    run_phase("W+", Wp_variable, config.global_lr_scale * 0.02)

                    Wpp_variable = WppVariable.from_Wp(Wp_variable)
                    run_phase("W++", Wpp_variable, config.global_lr_scale * 0.005)
                    
#this is our loss function on updated code of the loss function. gradient and feature one . i have all ready run baselin code with original cotton99_10 data. with same updated code.
#after running the code what you can do is put the updated code to this folder if you want to.