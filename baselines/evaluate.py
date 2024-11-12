from utils import cfg_parsing
from execution import testing_and_reporting



if __name__ == "__main__":
    cfg = cfg_parsing()
    testing_and_reporting(cfg)
    