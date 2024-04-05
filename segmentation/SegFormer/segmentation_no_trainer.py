import os
import math
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerConfig
from transformers import get_scheduler, default_data_collator
from dataset import SegmenDataset
import numpy as np
import json
import argparse
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
import evaluate



def parse_args():
    parser = argparse.ArgumentParser("Training Config")
    parser.add_argument("--pretrained_model_name_or_path", default = None, type = str)
    parser.add_argument("--dataset_dir", default = None, type= str)
    parser.add_argument("--label_dir", default=None, type = str)
    parser.add_argument("--output_dir", default = "/content/output", type = str)
    parser.add_argument("--cache_dir", type = str)

    #Model
    parser.add_argument("--reduce_labels", action= "store_true", 
                        help = " Reduce the background's label")
    parser.add_argument("--revision", default="main", type = str)
    

    #Trainer
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--lr_scheduler", default="constant", type = str,
                         help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            '                   "constant", "constant_with_warmup"]'))
    parser.add_argument("--batch_size", default = 2, type = int)
    parser.add_argument("--num_warmup_step", default = 0, type = int)
    parser.add_argument("--checkpointing_steps", default=None, type = str)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", default = 3, type = int)
    #parser.add_argument("--accumulate_grad_batches", default = 8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type = int)
    # parser.add_argument("--max_grad_norm", default=1.0, type = float)
    parser.add_argument("--weight_decay", default = 1e-2, type = float)
    parser.add_argument("--adam_beta1", default= 0.9, type = float)
    parser.add_argument("--adam_beta2", default= 0.999, type = float)
    parser.add_argument("--adam_epsilon", default=1e-8, type = float)   

    parser.add_argument("--report_to", default=None, type = str,
                        help = "Set report to wandb")
    parser.add_argument("--project_name", default= None, type = str)
    args = parser.parse_args()
    return args


def main():

    logger = get_logger(__name__)
    args = parse_args()
    js_path = os.path.join(args.dataset_dir, args.label_dir)

    
    with open (js_path, "r") as f:
        data = json.load(f)

    labels = data["annotations"]["meta"]["task"]["labels"]
    id2label = {i: labels["label"][i]["name"] for i in range(len(labels["label"]))}
    label2id = {v: k for k,v in id2label.items()}

    if args.report_to == "wandb":
        wandb.init(project = args.project_name)

    # Load model
    accelerator = Accelerator(gradient_accumulation_steps = args.gradient_accumulation_steps,
                              log_with= args.report_to)

    image_processor = SegformerImageProcessor.from_pretrained(args.pretrained_model_name_or_path, do_reduce_labels= args.reduce_labels)

    config = SegformerConfig.from_pretrained(args.pretrained_model_name_or_path, id2label = id2label, label2id = label2id)

    model = SegformerForSemanticSegmentation.from_pretrained(args.pretrained_model_name_or_path, config = config)
    
    logger.info(accelerator.state, main_process_only=False)
    
    # Dataset
    # check "validation folder"
    if os.path.isdir(os.path.join(args.dataset_dir, "val")) == False:
        if os.path.isdir(os.path.join(args.dataset_dir, "validation") == False):
            print("the directory must containing Tranining folder and Validation folder!!!")
            raise NotADirectoryError("The validation directory do not exist")
              
        else:
            val_data = SegmenDataset(path = args.dataset_dir, image_processor= image_processor, types = "validation")

    else:
        val_data = SegmenDataset(path = args.dataset_dir, image_processor= image_processor, types = "val")

    # Dataloader
    train_data = SegmenDataset(path = args.dataset_dir, image_processor = image_processor, types = "train")

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size, collate_fn = default_data_collator)

    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size = args.batch_size, collate_fn = default_data_collator)

    #Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()),
        lr = args.lr,
        betas = [args.adam_beta1, args.adam_beta2],
        eps = args.adam_epsilon,
        weight_decay = args.weight_decay,
    )

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    

    overrode_max_train_steps = False
    num_update_steps_per_epochs = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epochs
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        name = args.lr_scheduler,
        optimizer = optimizer,
        num_warmup_steps = args.num_warmup_step * accelerator.num_processes,
        num_training_steps = args.max_train_steps 
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    metric = evaluate.load("mean_iou", cache_dir = args.cache_dir)


    #Train
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_data)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.

    progress_bar = tqdm(range(args.max_train_steps), disable = not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch  = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    # progress_bar.update(completed_steps)
    progress_bar = tqdm(range(0, args.max_train_steps), disable = not accelerator.is_local_main_process, initial= completed_steps)
    progress_bar.set_description("Steps")

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        loss_dict = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # pixel_values = batch["pixel_values"]
                # labels = batch["labels"]
                # outputs = model(pixel_values = pixel_values, labels = labels)
                outputs = model(**batch)
                loss = outputs.loss
                loss_dict += loss * args.batch_size
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if accelerator.is_main_process:
                    logger.info("***** Running Eluavation *****")
                    #model.eval()

                    for step, batch in enumerate(val_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)

                            upsampled_logits = torch.nn.functional.interpolate(
                                outputs.logits, size = batch["labels"].shape[-2:], mode = "bilinear", align_corners=False
                            )
                            predicted = upsampled_logits.argmax(dim=1)
                            predictions, references = accelerator.gather_for_metrics((predicted, batch["labels"]))

                            metric.add_batch(
                                            predictions = predictions.detach().cpu().numpy(),
                                            references = references.detach().cpu().numpy(),
                            )

                            eval_metrics = metric.compute(
                                                        num_labels = len(id2label),
                                                        ignore_index = 255,  # ignore background
                                                        reduce_labels = False,
                            )
                            logger.info(f"epoch {epoch}: {eval_metrics}")
        
                            wandb.log(
                                    {
                                        "Eval_metic": eval_metrics
                            })
            
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{checkpointing_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = completed_steps)

            if completed_steps >= args.max_train_steps:
                break
            wandb.log({
            "Train_loss": loss.detach().item()
            })
        
    
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            unwarpped_model = accelerator.unwrap_model(model)
            unwarpped_model.save_pretrained(
                args.output_dir, is_main_process = accelerator.is_main_process, save_function = accelerator.save
                )

            image_processor.save_pretrained(args.output_dir)

        all_results = {
            f"eval_{k}": v.tolist() if isinstance(v, np.ndarray) else v for k, v in eval_metrics.items()
        }
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)   
    
    accelerator.end_training() 

if __name__ == "__main__":
    main()


# adding wandb
