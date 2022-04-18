import os
import logging

ROOT_PATH=os.path.join('/ssd2t/linqiubin/Datasets/dual_encoding')

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)