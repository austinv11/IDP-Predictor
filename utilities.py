import torch


# Move off the cpu when possible
from matplotlib import pyplot as plt


def move_off_cpu(obj, fallback_to_dml=True):
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
