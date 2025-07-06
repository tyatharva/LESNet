"""This package contains modules related to objective functions, optimizations, and network architectures.

To add a custom model class called 'dummy', you need to add a file called 'dummy_model.py' and define a subclass DummyModel inherited from BaseModel.
You need to implement the following five functions:
    -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    -- <set_input>:                     unpack data from dataset and apply preprocessing.
    -- <forward>:                       produce intermediate results.
    -- <optimize_parameters>:           calculate loss, gradients, and update network weights.
    -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.

In the function <__init__>, you need to define four lists:
    -- self.loss_names (str list):          specify the training losses that you want to plot and save.
    -- self.model_names (str list):         define networks used in our training.
    -- self.visual_names (str list):        specify the images that you want to display and save.
    -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an usage.

Now you can use the model class by specifying flag '--model dummy'.
See our template model class 'template_model.py' for more details.
"""

import importlib
from models.base_model import BaseModel
import torch
import torch.nn as nn


def find_model_using_name(model_name):
    """Import the module "models/[model_name]_model.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name + "_model"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def calc_weights(y_true, lake, a, b=24.0):
    exponents = {
        's': 19.0468,
        'm': 12.0062,
        'e': 16.0297,
        'o': 12.2432
    }
    k = exponents.get(lake, 14.2951)
    weights = (1 / k) * torch.exp(k * y_true)
    weights = torch.pow(weights, 1.0 / a) * (1.0 / a)
    max_weight = torch.tensor(b, device=y_true.device, dtype=y_true.dtype)
    return torch.minimum(weights, max_weight)


class WeightedL1Loss(nn.Module):
    def __init__(self, lake, a):
        super(WeightedL1Loss, self).__init__()
        self.lake = lake
        self.a = a

    def forward(self, y_pred, y_true):
        l1_loss = torch.abs(y_pred - y_true)
        weighted_loss = calc_weights(y_true, self.lake, self.a) * l1_loss
        return weighted_loss.mean()


class WeightedL2Loss(nn.Module):
    def __init__(self, lake, a):
        super(WeightedL2Loss, self).__init__()
        self.lake = lake
        self.a = a

    def forward(self, y_pred, y_true):
        l2_loss = torch.square(y_pred - y_true)
        weighted_loss = calc_weights(y_true, self.lake, self.a) * l2_loss
        return weighted_loss.mean()


class LossFunctionSelector:
    def __init__(self):
        self.loss_functions = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
            'wl1': WeightedL1Loss,
            'wl2': WeightedL2Loss,
        }

    def get_loss_function(self, opt):
        """
        Returns the corresponding loss function based on the input string.
        
        Args:
            opt: The options.
        
        Returns:
            torch.nn.Module: The corresponding loss function.
        
        Raises:
            ValueError: If the loss type is not recognized.
        """
        if opt.loss_fn.lower() == 'wl1': return WeightedL1Loss(opt.dataroot[-1], opt.loss_a)
        elif opt.loss_fn.lower() == 'wl2': return WeightedL2Loss(opt.dataroot[-1], opt.loss_a)
        elif opt.loss_fn.lower() in self.loss_functions: return self.loss_functions[opt.loss_fn.lower()]
        else: raise ValueError(f"Loss type '{opt.loss_fn}' is not recognized.")
