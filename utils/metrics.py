import torch


def calc_miou(pred_batch, label_batch, num_classes, ignore_index):
    '''
    :param pred_batch: [b,h,w]
    :param label_batch: [b,h,w]
    :param num_classes: scalar
    :param ignore_index: scalar
    :return:
    '''
    miou_sum, miou_count = 0, 0
    for batch_idx in range(label_batch.shape[0]):
        pred, label = pred_batch[batch_idx].flatten(0), label_batch[batch_idx].flatten(0)

        mask = label != ignore_index
        pred, label = pred[mask], label[mask]

        pred_one_hot = torch.nn.functional.one_hot(pred, num_classes)
        label_one_hot = torch.nn.functional.one_hot(label, num_classes)

        intersection = torch.sum(pred_one_hot * label_one_hot)
        union = torch.sum(pred_one_hot) + torch.sum(label_one_hot) - intersection + 1e-6

        miou_sum += intersection / union
        miou_count += 1
    return miou_sum / miou_count
