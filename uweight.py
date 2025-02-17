import torch
import torch.nn as nn

def uncertainty_to_weigh_losses(loss_list):
    """
    所有任务的损失列表，为tensor格式
    :param loss_list:
    :return: tensor格式的综合损失
    """
    loss_n = len(loss_list)
    uncertainty_weight = [
        nn.Parameter(torch.tensor([1 / loss_n]), requires_grad=True) for _ in range(loss_n)
    ]

    final_loss = []
    for i in range(loss_n):
        final_loss.append(loss_list[i] / (2 * uncertainty_weight[i]**2) + torch.log(uncertainty_weight[i]))

    return sum(final_loss)

