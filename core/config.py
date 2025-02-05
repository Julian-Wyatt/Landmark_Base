import yaml


class SubConfig:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def update(self, entries: dict):
        for key, value in entries.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not found in config")

    def __str__(self):
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


class Dataset(SubConfig):
    def __init__(self, **entries):
        self.ROOT_DIR = ""
        self.NAME = ""
        self.IMG_DIRS = [""]
        self.IMG_EXT = ""
        self.ANNOTATIONS_FILE = ""
        self.PARTITION_FILE = ""
        self.CACHE_DIR = ""

        # Image-related attributes
        self.NUMBER_KEY_POINTS = 0
        self.CHANNELS = 3
        self.PIXELS_PER_MM = [1, 1]
        self.IMG_SIZE = [128, 128]

        self.GT_SIGMA = 1

        # Training-related attributes
        self.AUGMENT_TRAIN = True

        # int to float methods:
        # 0-255 (none), image/255 (standard), max-min/max (0-1), adaptive
        # normalisation methods:
        # 0-1 (none), -1-1 (mu=0.5, sigma=0.5), dataset mean and std

        self.INT_TO_FLOAT = "none"  # ["none", "standard", "minmax", "adaptive"]
        self.NORMALISATION = "none"  # ["none", "mu=0.5,sig=0.5", "dataset"]

        super().__init__(**entries)


class Augmentations(SubConfig):
    def __init__(self, **entries):
        # refined from challenge
        # AUGMENTATIONS:
        # INVERT_RATE: 0.1  # between 0 and 0.15 is best
        # BLUR_RATE: 0.1  # between 0 and 0.2 is best
        # ROTATION: 5  # between 0 and 8 is best
        # SCALE: 0.125  # 1.25 far best which was top of sweep so trying 1.5
        # SHEAR: 0  # Fairly no correlation but around 5 gives decent results
        # TRANSLATION_X: [-30, 30]  # between 20 and 30 is best
        # TRANSLATION_Y: [-20, 20]  # lowerbound between [-20,-25] is best Upper bound between 20 and 30
        # MULTIPLY: 0.4  # Fairly no correlation - about 0.45 is peak but best result at 0.65
        # CUTOUT_ITERATIONS: 1  # 1 for higher sizes or 2 for smaller sizes is best
        # CUTOUT_SIZE_MIN: 0.04  # 0.07-0.1 is best
        # CUTOUT_SIZE_MAX: 0.3  # 0.25-0.3 is best
        # GAUSSIAN_NOISE: 0  # 0 is best
        # USE_SKEWED_SCALE_RATE: 0.3  # 0.25 is best although little correlation
        # CONTRAST_GAMMA_MIN: 0.3  # 0.3 is best roughly - lower values are better
        # CONTRAST_GAMMA_MAX: 2  # 2.0 is best roughly - 2-2.2 values are better
        # SIMULATE_XRAY_ARTEFACTS_RATE: 0.9  # 0.7-0.9 is best
        # ELASTIC_TRANSFORM_ALPHA: 400
        # ELASTIC_TRANSFORM_SIGMA: 30
        self.ROTATION = 5
        self.SCALE = 0.125
        self.TRANSLATION_X = [-20, 20]
        self.TRANSLATION_Y = [-20, 20]
        self.SHEAR = 0
        self.MULTIPLY = 0.4
        self.GAUSSIAN_NOISE = 0.02
        self.ELASTIC_TRANSFORM_ALPHA = 400
        self.ELASTIC_TRANSFORM_SIGMA = 30
        self.COARSE_DROPOUT_RATE = 0
        self.ADDATIVE_GAUSSIAN_NOISE_RATE = 0
        self.FLIP_INITIAL_COORDINATES = False
        self.CHANNEL_DROPOUT = 0
        self.CUTOUT_ITERATIONS = 1
        self.CUTOUT_SIZE_MIN = 0.04
        self.CUTOUT_SIZE_MAX = 0.3
        self.BLUR_RATE = 0.1
        self.CONTRAST_GAMMA_MIN = 0.3
        self.CONTRAST_GAMMA_MAX = 2
        self.SHARPEN_ALPHA_MIN = 0
        self.SHARPEN_ALPHA_MAX = 0.6
        self.INVERT_RATE = 0
        self.USE_SKEWED_SCALE_RATE = 0.1
        self.SIMULATE_XRAY_ARTEFACTS_RATE = 0.9

        super().__init__(**entries)


class Model(SubConfig):
    # class conditional timestep unet with attention
    def __init__(self, **entries):
        self.NAME = "Image Unet"
        self.DROPOUT = 0.1
        super().__init__(**entries)


class TrainLosses(SubConfig):
    def __init__(self, **entries):
        self.NLL_WEIGHT = 0
        self.BCE_WEIGHT = 0
        # self.OFFSET_LOSS_WEIGHT = 0

        self.MASK_RADIUS = 0
        # self.USE_OFFSETS = False
        # self.LOCALISED_LOSS = False

        super().__init__(**entries)


class Train(SubConfig):
    def __init__(self, **entries):
        self.BATCH_SIZE = 1
        self.LR = 0.01
        self.EPOCHS = 10
        self.NUM_WORKERS = 4
        self.WEIGHT_DECAY = 0.0
        self.BETA1 = 0.9
        self.BETA2 = 0.999
        self.OPTIMISER = "adamw"
        self.LOG_IMAGE = False
        self.MODEL_TYPE = "default"
        self.SAVING_ROOT_DIR = ""
        self.LOG_WHOLE_VAL = False
        self.DEBUG = False
        self.DESCRIPTION = ""
        self.USE_SCHEDULER = False
        self.RUN_TEST = True
        self.RUN_TRAIN = True
        self.LOG_TEST_METRICS = True
        self.CHECKPOINT_FILE = ""
        self.TOP_K_HOTTEST_POINTS = 10

        self.VAL_EVERY_N_EPOCHS = 20

        self.MIN_LR = 1e-6
        self.WARMUP_EPOCHS = 15
        self.EARLY_STOPPING_WARMUP = 15

        self.PROJECT = "fine-tune"  # wandb project name

        self.EXP_LR_DECAY = 0.94

        self.ACCUMULATOR = {0: 64, 5: 32, 10: 16, 15: 8}
        super().__init__(**entries)


class Config:
    def __init__(self, **entries):
        self.DATASET = Dataset(**entries.get("DATASET", {}))
        self.MODEL = Model(**entries.get("MODEL", {}))
        self.AUGMENTATIONS = Augmentations(**entries.get("AUGMENTATIONS", {}))
        self.TRAIN = Train(**entries.get("TRAIN", {}))
        self.TRAINLOSSES = TrainLosses(**entries.get("TRAINLOSSES", {}))
        self.PATH = ""

    def __str__(self):
        sections = ["DATASET", "MODEL", "AUGMENTATIONS", "TRAIN", "TRAINLOSSES"]
        output = ""
        for section in sections:
            output += f"{section}\n"
            section_obj = getattr(self, section)
            output += "\n".join([f"\t{key}: {value}" for key, value in section_obj.__dict__.items()])
            output += "\n\n"
        return output

    def to_dict(self):
        return {section: getattr(self, section).__dict__ for section in
                ["DATASET", "MODEL", "AUGMENTATIONS", "TRAIN", "TRAINLOSSES"]}

    def __copy__(self):
        return Config(**self.to_dict())

    def update(self, entries: dict):
        for key, value in entries.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise ValueError(f"Key {key} not found in config")

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)


def get_config(cfg_path, saving_root_dir="./") -> Config:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    _cfg = Config(**cfg)
    _cfg.PATH = cfg_path
    _cfg.TRAIN.SAVING_ROOT_DIR = saving_root_dir
    if _cfg.DATASET.ROOT_DIR.startswith("/data/") and _cfg.DATASET.CACHE_DIR == "":
        _cfg.DATASET.CACHE_DIR = f"{'/'.join(_cfg.TRAIN.SAVING_ROOT_DIR.split('/')[:-1])}/dataset_cache/"
    elif _cfg.DATASET.CACHE_DIR == "":
        _cfg.DATASET.CACHE_DIR = "../datasets/dataset_cache"
    return _cfg


if __name__ == "__main__":
    # cfg = get_config("./configs/default.yaml")
    # print(get_config("./configs/autoencoder.yaml"))
    print(get_config("../configs/local_test_ceph_ISBI.yaml"))
