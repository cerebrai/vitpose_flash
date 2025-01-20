from typing import List
import cv2
from tqdm import tqdm
import numpy as np
import torch

import vitpose_flash.onepose as onepose
from vitpose_flash.onepose.models.factory import Model

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def get_runtime(model: Model, inputs: List, num_iter: int = 100) -> float:
    """ Calculates the average runtime of passing the inputs through
    the model, averared over the specified number of iterations

    Args:
        model: vitpose model defined in vitpose_flash.onepose.models.factory.Model
        inputs: A list of images, the length of the list is equal to the batch_size.
        num_iter: the number of iterations to find the average runtime.

    Returns:
        average runtime


    """
    # -- Warming up
    print("Warming up")
    for k in range(2):
        model(inputs)
    print("Warming up is done")

    # -- Calculating average runtime
    time_array = np.zeros(num_iter)
    for k in tqdm(range(num_iter)):
            _, time_array[k] = timed(lambda: model(inputs))

    return float(np.mean(time_array))

def benchmark(inputs: List):
    """ Computes and prints the runtime for various vitpose flash
    configurations, for given inputs
    """
    print(f"# {'-' * 10} Benchmarking for batch size  = {len(inputs)} {'-' * 10} #")

    # --- Standard VITPose runtime ---#
    print("# -- Vitpose+ Large Standard model")
    model = onepose.create_model('ViTPose+_large_coco_wholebody').cuda()
    runtime = get_runtime(model, inputs)
    print(f"Average runtime of the standard VitPose Large: {runtime * 1000} ms")
    print(f"# {'-' * 40}")

    # --- VITPose runtime with float16 precision---#
    print("# -- Vitpose+ Large with float16 precision")
    model = onepose.create_model('ViTPose+_large_coco_wholebody',
                                dtype=torch.float16).cuda()
    runtime = get_runtime(model, inputs)
    print(f"Average runtime of the VitPose Large with float16 precision: {runtime * 1000} ms")
    print(f"# {'-' * 40}")

    # --- VITPose runtime with float16 precision and flash attention---#
    print("# -- Vitpose+ Large with float16 precision and flash attention")
    model = onepose.create_model('ViTPose+_large_coco_wholebody',
                                dtype=torch.float16,
                                flash_attention=True).cuda()
    runtime = get_runtime(model, inputs)
    print(f"Average runtime of the VitPose Large with float16 precision and flash attention: {runtime * 1000} ms")
    print(f"# {'-' * 40}")

    # --- VITPose runtime with float16 precision, flash attention and TensorRT compile---#
    print("# -- Vitpose+ Large with float16 precision, flash attention and TensorRT compile")
    model = onepose.create_model('ViTPose+_large_coco_wholebody',
                                dtype=torch.float16,
                                flash_attention=True,
                                compile_flag=True,
                                compile_batch_size=len(inputs)).cuda()
    runtime = get_runtime(model, inputs)
    print(f"Average runtime of the VitPose Large with float16 precision, flash attention and TensorRT compile: {runtime * 1000} ms")
    print(f"# {'-' * 40}")


def main():
    """ Main function
    """

    test_image = cv2.imread("sample.png")

    # -- Benchmark for batch_size equals to 1
    batch_size = 1
    inputs = [test_image for _ in range(batch_size)]
    benchmark(inputs)

    # -- Benchmark for batch_size equals to 32
    batch_size = 32
    inputs = [test_image for _ in range(batch_size)]
    benchmark(inputs)



if __name__ == "__main__":

    main()
