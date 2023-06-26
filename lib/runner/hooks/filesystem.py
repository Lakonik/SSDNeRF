from distutils.dir_util import copy_tree

from mmcv.runner import HOOKS, Hook, get_dist_info


@HOOKS.register_module()
class DirCopyHook(Hook):
    def __init__(self,
                 interval=-1,
                 by_epoch=False,
                 in_dir=None,
                 out_dir=None,
                 save_last=True):
        self.interval = interval
        self.by_epoch = by_epoch
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.save_last = save_last

    def _backup(self):
        rank, ws = get_dist_info()
        if rank == 0:
            copy_tree(self.in_dir, self.out_dir)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        if self.every_n_epochs(
                runner, self.interval) or (self.save_last
                                           and self.is_last_epoch(runner)):
            runner.logger.info(
                f'Backing up cache files at {runner.epoch + 1} epochs')
            self._backup()

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        if self.every_n_iters(
                runner, self.interval) or (self.save_last
                                           and self.is_last_iter(runner)):
            runner.logger.info(
                f'Backing up cache files at {runner.iter + 1} iterations')
            self._backup()
