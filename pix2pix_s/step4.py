from subprocess import call, DEVNULL
from tqdm import tqdm
to_do = [('0.001', 'l1', '50', '24.0', '14'),
         ('0.001', 'wl1', '50', '24.0', '14'),]

total_iterations = len(to_do)

with open('err.txt', 'w') as err_file:
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for lr, lfn, l1, la, varn in to_do:
            call([
                "python",
                "train.py",
                "--dataroot", "../dataset/s",
                "--name", f"{lr}_{lfn}_{l1}_{la}_{varn}_s",
                "--model", "pix2pix",
                "--input_nc", f"{varn}",
                "--output_nc", "1",
                "--norm", "batch",
                "--init_type", "kaiming",
                "--dataset_mode", "lake",
                "--num_threads", "16",
                "--batch_size", "8",
                "--n_epochs", "25",
                "--n_epochs_decay", "0",
                "--save_epoch_freq", "55",
                "--netG", "unetformer",
                "--netD", "spectral",
                "--lr_policy", "cosine",
                "--lr", f"{lr}",
                "--lambda_L1", f"{l1}",
                "--loss_fn", f"{lfn}",
                "--loss_a", f"{la}",
                "--gan_mode", "lsgan",
                "--phase", "train",
            ], stdout=DEVNULL, stderr=err_file)

            call([
                "python",
                "test.py",
                "--dataroot", "../dataset/s",
                "--name", f"{lr}_{lfn}_{l1}_{la}_{varn}_s",
                "--model", "pix2pix",
                "--input_nc", f"{varn}",
                "--output_nc", "1",
                "--init_type", "xavier",
                "--dataset_mode", "lake",
                "--num_threads", "16",
                "--num_test", "336",
                "--netG", "unetformer",
                "--phase", "test",
                "--eval", "--saliency",
            ], stdout=DEVNULL, stderr=err_file)
        
            pbar.update(1)
