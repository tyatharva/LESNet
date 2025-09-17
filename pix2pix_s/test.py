"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import pandas as pd
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
from torch.nn.functional import l1_loss
import numpy as np
from collections import defaultdict

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    # optional: wandb logging
    if opt.use_wandb:
        wandb_run = wandb.init(
            project=opt.wandb_project_name,
            name=opt.name,
            config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # make output HTML
    web_dir = os.path.join(opt.results_dir, opt.name,
                           f'{opt.phase}_{opt.epoch}')
    if opt.load_iter > 0:
        web_dir = f'{web_dir}_iter{opt.load_iter}'
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir,
                       f'Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}')
    if opt.eval:
        model.eval()

    # metrics accumulators
    model_metrics_accum = None
    if opt.phase == 'val':
        if opt.full_val:
            model_metrics_accum = dict.fromkeys(
                ['MSE','MAE','PCC','MS-SSIM',
                 'FSS_0.25_5','FSS_1_5','FSS_2_5'], 0.0)
        else:
            model_metrics_accum =  dict.fromkeys(['Combined'], 0.0)
    else:
        model_metrics_accum = dict.fromkeys(
            ['MSE','MAE','PCC','MS-SSIM',
             'FSS_0.25_5','FSS_1_5','FSS_2_5'], 0.0)
    hrrr_metrics_accum = model_metrics_accum.copy()
    num_images = 0

    # prepare Excel writer
    excel_path = os.path.join(opt.results_dir, 'metrics.xlsx')
    if os.path.exists(excel_path):
        writer = pd.ExcelWriter(excel_path, engine='openpyxl',
                                mode='a', if_sheet_exists='overlay')
    else:
        writer = pd.ExcelWriter(excel_path, engine='openpyxl', mode='w')

    # --- prepare saliency accumulation ---
    num_ch = opt.input_nc
    saliency_sum = np.zeros(num_ch, dtype=np.float64)
    batch_count  = 0

    # prepare bias ratio calculation
    mb_sum = defaultdict(lambda: [0, 0])
    hb_sum = defaultdict(lambda: [0, 0])

    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print(f'processing ({i:04d})-th image... {img_path}')
        model_metrics, hrrr_metrics, model_bias, hrrr_bias = save_images(
            opt.phase, opt.full_val, webpage, visuals,
            img_path, aspect_ratio=opt.aspect_ratio,
            width=opt.display_winsize, use_wandb=opt.use_wandb)

        # accumulate metrics
        for metric in model_metrics_accum:
            model_metrics_accum[metric] += model_metrics[metric]
            hrrr_metrics_accum[metric] += hrrr_metrics[metric]
        num_images += 1

        if opt.saliency:
            # ---- compute & accumulate saliency for this batch ----
            input_A  = data['A'].to(model.device)
            target_B = data['B'].to(model.device)
            if input_A.grad is not None:
                input_A.grad.zero_()
            model.netG.zero_grad()

            # enable gradients on the inputs
            input_A.requires_grad_(True)

            # forward through scaler + generator
            # scaled_A = model.input_scaler(input_A)
            fake_B   = model.netG(input_A)

            # use sum of the output as a dummy scalar loss
            loss = fake_B.sum()
            loss.backward()

            # mean absolute gradient over (batch,H,W) → per-channel
            sal = input_A.grad.abs().mean(dim=[0,2,3]).cpu().numpy()
            saliency_sum += sal
            batch_count   += 1

            # add to bias ratio counting
            for k, (ps, ts) in model_bias.items():
                mb_sum[k][0] += ps  # add pred_sum
                mb_sum[k][1] += ts  # add target_sum
            for k, (ps, ts) in hrrr_bias.items():
                hb_sum[k][0] += ps  # add pred_sum
                hb_sum[k][1] += ts  # add target_sum

            # clear gradients
            input_A.grad.zero_()
            model.netG.zero_grad()

    # average metrics
    for metric in model_metrics_accum:
        model_metrics_accum[metric] /= num_images
        hrrr_metrics_accum[metric]  /= num_images

    # write metrics to Excel
    for metric in model_metrics_accum:
        df_new = pd.DataFrame({
            'model': [model_metrics_accum[metric]],
            'hrrr' : [hrrr_metrics_accum[metric]]
        }, index=[opt.name])
        sheet = metric
        if sheet in writer.sheets:
            startrow = writer.sheets[sheet].max_row
            df_new.to_excel(writer, sheet_name=sheet,
                            startrow=startrow,
                            header=False, index=True)
        else:
            df_new.to_excel(writer, sheet_name=sheet, index=True)

    # ---- compute & save average saliency ranking ----
    if opt.saliency:
        saliency_mean = saliency_sum / batch_count
        sal_path = os.path.join(web_dir, 'saliency.txt')
        with open(sal_path, 'w') as f:
            f.write('\n=== Average Saliency per Channel over Test Split ===\n')
            for idx in range(num_ch):
                f.write(f'Channel {idx:2d} — Saliency: {saliency_mean[idx]:.6f}\n')

    # Average and save out bias ratios
    model_ratios = {k: (ps / ts if ts > 0 else np.nan) for k, (ps, ts) in mb_sum.items()}
    hrrr_ratios = {k: (ps / ts if ts > 0 else np.nan) for k, (ps, ts) in hb_sum.items()}
    bias_path = os.path.join(web_dir, 'bias_ratios.txt')
    with open(bias_path, 'w') as f:
        f.write("MODEL\n")
        f.write(f"{'Bucket':<15}{'Bias Ratio':>12}\n")
        f.write("-" * 27 + "\n")
        for (low, high), ratio in model_ratios.items():
            if high == np.inf:
                bucket = f">{low}"
            else:
                bucket = f"{low}–{high}"
            f.write(f"{bucket:<15}{ratio:>12.6f}\n")

        f.write("\n\nHRRR\n")
        f.write(f"{'Bucket':<15}{'Bias Ratio':>12}\n")
        f.write("-" * 27 + "\n")
        for (low, high), ratio in hrrr_ratios.items():
            if high == np.inf:
                bucket = f">{low}"
            else:
                bucket = f"{low}–{high}"
            f.write(f"{bucket:<15}{ratio:>12.6f}\n")



    writer._save()
    webpage.save()
