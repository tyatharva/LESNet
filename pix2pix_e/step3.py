from subprocess import call, DEVNULL
from tqdm import tqdm
to_do = [('0.001', 'wl1', '50', '14'),
         ('0.001', 'wl1', '25', '9'),
         ('0.0005', 'wl1', '50', '9'),
         ('0.001', 'wl1', '50', '7'),
         ('0.001', 'wl1', '25', '14')]
las = [12.0, 18.0, 24.0]
total_iterations = len(to_do) * len(las)

with open('err.txt', 'w') as err_file:
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for lr, lfn, l1, varn in to_do:
            for la in las:
                call([
                    "python",
                    "train.py",
                    "--dataroot", "../dataset/e",
                    "--name", f"{lr}_{lfn}_{l1}_{la}_{varn}_e",
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
                    "--dataroot", "../dataset/e",
                    "--name", f"{lr}_{lfn}_{l1}_{la}_{varn}_e",
                    "--model", "pix2pix",
                    "--input_nc", f"{varn}",
                    "--output_nc", "1",
                    "--init_type", "xavier",
                    "--dataset_mode", "lake",
                    "--num_threads", "16",
                    "--num_test", "336",
                    "--netG", "unetformer",
                    "--phase", "val",
                    "--eval", "--full_val",
                ], stdout=DEVNULL, stderr=err_file)
            
                pbar.update(1)
