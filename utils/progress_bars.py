from lightning.pytorch.callbacks import TQDMProgressBar
from tqdm import tqdm


class noVal_testProgressBar(TQDMProgressBar):
    """https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning"""

    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,
        )
        return bar

    def init_test_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for testing. """
        bar = tqdm(
            disable=True,
        )
        return bar


class noProgressBar(noVal_testProgressBar):
    """https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning"""

    def init_train_tqdm(self) -> tqdm:
        """ Override this to customize the tqdm bar for training. """
        bar = tqdm(
            disable=True,
        )
        return bar
