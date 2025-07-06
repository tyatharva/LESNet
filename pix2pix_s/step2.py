from subprocess import call, DEVNULL
from tqdm import tqdm

lrs = [0.0005, 0.001]
lfns = ['wl1']
l1s = [5, 25, 50]
las = [18.0]
varns = [2, 7, 9, 10, 14]
total_iterations = len(lrs) * len(lfns) * len(l1s) * len(las) * len(varns)

with open('err.txt', 'w') as err_file:
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for lr in lrs:
            for lfn in lfns:
                for l1 in l1s:
                    for la in las:
                        for varn in varns:
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
                                "--save_epoch_freq", "50",
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
                                "--phase", "val",
                                "--eval",
                            ], stdout=DEVNULL, stderr=err_file)
                
                            pbar.update(1)
