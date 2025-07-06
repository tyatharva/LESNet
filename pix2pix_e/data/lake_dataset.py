import os
import zarr
import glob
import torch
from data.base_dataset import BaseDataset


class LakeDataset(BaseDataset):
    """A dataset class for paired image dataset.

    Modified to work with the LES dataset
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = self._get_sample_paths()  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def _get_sample_paths(self):
        sample_paths = []
        input_paths = sorted(glob.glob(f"{self.dir_AB}A/*input.zarr"))
        for input_path in input_paths:
            target_path = input_path.replace("A", "B").replace("input.zarr", "target.zarr")
            if os.path.exists(target_path):
                sample_paths.append((input_path, target_path))
        return sorted(sample_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - input tensor
            B (tensor) - - output tensor
            A_paths (str) - - input zarr store path
            B_paths (str) - - target zarr store path
        """
        # read a image given a random integer index
        A_paths, B_paths = self.AB_paths[index]
        A_store = zarr.open(A_paths, mode='r')
        B_store = zarr.open(B_paths, mode='r')
        A_vars = []
        #                                           0            1            2             3             4            5            6             7            8             9            10           11       12        13
        if (self.opt.input_nc == 14): A_vars = ['QPE_past', 'SHSR_mrms', 'UGRD_850mb', 'VGRD_850mb', 'DPT_850mb', 'TMP_850mb', 'UGRD_925mb', 'VGRD_925mb', 'DPT_925mb', 'TMP_925mb', 'TMP_surface', 'DPT_2m', 'elev', 'landsea'] # Gowan paper
        elif (self.opt.input_nc == 10): A_vars = ['QPE_past', 'SHSR_mrms', 'CAPE_surface', 'TMP_masked', 'TMP_850mb', 'DPT_850mb', 'UGRD_850mb', 'VGRD_850mb', 'ICEC_surface', 'elev'] # My idea #1
        #                                            0            1            2              3             4             5             6             7           8
        elif (self.opt.input_nc == 9): A_vars = ['QPE_past', 'SHSR_mrms', 'THTE_masked', 'THTE_850mb', 'UGRD_850mb', 'VGRD_850mb', 'DIVG_925mb', 'RELV_925mb', 'flow'] # My idea #2
        elif (self.opt.input_nc == 7): A_vars = ['QPE_past', 'SHSR_mrms', 'TMP_surface', 'TMP_850mb', 'UGRD_850mb', 'VGRD_850mb', 'elev'] # Conventional
        elif (self.opt.input_nc == 2): A_vars = ['QPE_past', 'SHSR_mrms'] # Bare minimum

        A = torch.stack([torch.from_numpy(A_store[var][:, :]) for var in A_vars], dim=0).float() # Shape: (c, 256, 512) or (c, 512, 256) for Lake Michigan
        B = torch.from_numpy(B_store['QPE_target'][:, :]).unsqueeze(0).float() # Shape: (1, 256, 512) or (1, 512, 256) for Lake Michigan

        A = torch.flip(A, dims=[1]) # Flip over the x-axis to get the actual true image/data
        B = torch.flip(B, dims=[1])

        if torch.isnan(A).any(): print(f"NAN in INPUT {A_paths}")
        if torch.isnan(B).any(): print(f"NAN in TARGET{B_paths}")

        return {'A': A, 'B': B, 'A_paths': A_paths, 'B_paths': B_paths}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
