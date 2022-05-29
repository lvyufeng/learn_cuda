import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")


if __name__ == "__main__":
    # 定义aot类型的自定义算子
    def infer_shape(x):
        return [1,]
    op = ops.Custom("./reduction_shared_memory.so:ReductionNative", out_shape=infer_shape, out_dtype=lambda x: x, func_type="aot")

    x0 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).astype(np.float32)
    # x0 = np.array([0, 1, 2, 3, 4, 5, 6, 7]).astype(np.float32)
    """
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    2, 3, 4, 5, 6, 7, 8, 9, 10
    5, 7, 9, 11, 13, 15, 19
    10, 13, 16, 19, 22, 25
    19, 23
    """
    output = op(Tensor(x0))
    print(output)