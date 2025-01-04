import cv2
from tqdm import tqdm
import numpy as np
import torch

import vitpose_flash.onepose as onepose

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


if __name__ == "__main__":

    test_image = cv2.imread("/home/safwan/gsplat/test_data/tennis_output_frames/0_new/0/im_1.png")
    model = onepose.create_model('ViTPose+_large_coco_wholebody').to("cuda")
    iter = 100
    time_array = np.zeros(iter)
    for k in tqdm(range(iter)):
            time_array[k] = timed(lambda: model(test_image))[1]

    print(f"Average Time: {np.mean(time_array)}" )
