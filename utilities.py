import torch
import torch.nn.functional as F


# Move off the cpu when possible
from matplotlib import pyplot as plt


def move_off_cpu(obj, fallback_to_dml=False):
    if torch.cuda.is_available():
        # Try to move to CUDA
        return obj.to('cuda')
    # Try to move to Microsoft DirectML to run on AMD GPUs
    elif fallback_to_dml:
        try:
            # Try to move to Microsoft DirectML if available
            return obj.to('dml')
        except:
            pass
    else:
        try:
            # Try to move to TPU (i.e. on Google Colab)
            import torch_xla.utils.utils as xu
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return obj.to(device)
        except:
            pass
    return obj


# Given losses and accuracy per epoch for models,
# make a plot comparing the curves
def plot_loss_accuracy(epoch2losses, epoch2accs,
                       names, plot_name):
    # Plot loss and accuracy
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"{names[0]+': ' if len(names) == 1 else ''}"
                 f"{plot_name} Loss and Accuracy")
    plt.subplot(1, 2, 1)
    plt.title("Binary Cross-Entropy Loss")
    for i, epoch2loss in enumerate(epoch2losses):
        plt.plot(epoch2loss, label=names[i])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if len(names) > 1:
        plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Binary Accuracy")
    for i, epoch2acc in enumerate(epoch2accs):
        plt.plot(epoch2acc, label=names[i])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if len(names) > 1:
        plt.legend()
    plt.show()
    plt.clf()


def sliding_window(tensor, window_size, dimension=1, stride=1, flatten=True, centered=True):
    """
    Given a tensor, return a tensor of sliding windows centered on each position
    https://stackoverflow.com/a/53972525/5179044
    """
    assert window_size % 2 == 1, "Window size must be odd"

    # Since the window is centered on each position, we must pad the tensor
    # with zeros to the left and right
    padding = (window_size - 1) // (2 if centered else 1)

    # Pad the tensor with zeros to the left and right
    dims = list(tensor.size())
    dims[dimension] = padding
    if padding == 0:
        padded = tensor
    else:
        zeros = tensor.new_zeros(dims)
        padded = torch.cat([zeros, tensor, zeros], dim=dimension)

    windows = padded.unfold(dimension, window_size, stride)
    if flatten:
        # Flatten the window dimension into the feature dimension
        windows = windows.flatten(len(dims)-1)
    return windows
