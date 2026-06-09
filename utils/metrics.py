import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, n_classes=10, n_years=26, device='cpu'):
        self.n_classes = n_classes
        self.n_years = n_years
        self.device = device
        self.reset()
        
    def reset(self):
        self.count = 0
        self.loss = 0
        self.acc = 0
        self.mae = 0
        self.signed_mae = 0
        self.acc1 = 0   
        self.acc2 = 0
        self.conf_mat_frame_id = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int64, device=self.device)
        self.conf_mat_year = torch.zeros((self.n_years, self.n_years), dtype=torch.int64, device=self.device)

    def update(self, loss, pred, label, years):
        self.count += pred.size(0)
        self.loss += loss
        self.mae += abs(pred - label).sum()
        self.signed_mae += (pred - label).sum()
        self.acc += (pred == label).sum()
        self.acc1 += (abs(pred - label) <= 1).sum()
        self.acc2 += (abs(pred - label) <= 2).sum()
        self.conf_mat_frame_id += torch.bincount(label.flatten() * self.n_classes + pred.flatten(), minlength=self.n_classes**2).reshape(self.n_classes, self.n_classes)
        pred_years = years[torch.arange(pred.size(0), device=pred.device), pred] - 2000
        label_years = years[torch.arange(label.size(0), device=label.device), label] - 2000        
        self.conf_mat_year += torch.bincount(label_years.flatten() * self.n_years + pred_years.flatten(), minlength=self.n_years**2).reshape(self.n_years, self.n_years)

    def get_metrics(self):
        loss = self.loss / self.count
        mae = self.mae / self.count
        signed_mae = self.signed_mae / self.count
        acc = self.acc / self.count
        acc1 = self.acc1 / self.count
        acc2 = self.acc2 / self.count
        return loss.item(), mae.item(), signed_mae.item(), acc.item(), acc1.item(), acc2.item()