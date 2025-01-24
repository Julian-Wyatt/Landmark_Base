from lightning.pytorch.callbacks import EarlyStopping
import torch


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    EarlyStopping, except don't watch the first `warmup` epochs.
    """

    def __init__(self, warmup=10, **kwargs):
        super().__init__(**kwargs)
        self.warmup = warmup

    def on_validation_end(self, trainer, pl_module):

        if trainer.current_epoch < self.warmup:
            self.wait_count = 0
            return
        else:
            self._run_early_stopping_check(trainer)
