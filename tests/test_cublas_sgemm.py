import numpy as np
from mindspore import context, Tensor
import mindspore.ops as ops

context.set_context(device_target="GPU")

if __name__ == "__main__":
    def infer_shape(x_shape, y_shape):
        print(x_shape)
        return (x_shape[1], y_shape[0])

    # 定义aot类型的自定义算子
    op = ops.Custom("../build/cublas_sgemm.so:SGEMM", out_shape=infer_shape, out_dtype=lambda x, _: x, func_type="aot")

    x0 = np.random.randn(3, 2).astype(np.float32)
    x1 = np.random.randn(4, 3).astype(np.float32)
    output = op(Tensor(x0), Tensor(x1))
    print(output)