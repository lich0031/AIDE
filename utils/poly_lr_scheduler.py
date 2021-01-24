from torch.optim.optimizer import Optimizer

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class PolyLR(_LRScheduler):
    """ Learning rate policy where current learning rate of each
    parameter group equals to the initial learning rate multiplying
    by :math:`(1 - \frac{last_epoch % max_epoch}{max_epoch})^power`.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_epoch (int): Period of learning rate decay.
        power (float): Power factor of learning rate multiplier decay. Default: 0.9.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, max_epoch, power=0.9, last_epoch=-1):
        self.max_epoch = max_epoch
        self.power = power
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr_list = []
        for base_lr_ in self.base_lrs:
            lr_ = base_lr_ * ((1.0 - float(self.last_epoch % self.max_epoch) / float(self.max_epoch)) ** self.power)
            lr_list.append(lr_)
        return lr_list