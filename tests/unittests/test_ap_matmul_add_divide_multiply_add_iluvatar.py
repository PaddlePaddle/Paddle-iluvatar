# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest
import numpy as np
from pathlib import Path

import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct
import paddle.profiler as profiler


os.environ["AP_WORKSPACE_DIR"] = "/tmp/paddle/ap_workspace"


def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)


DT = "float16"
BS = 8 # 1, 8, 32
MS = 784
NS = 192
KS = 768


class TestMatmulAddDivideMultiplyAdd(unittest.TestCase):
    def setUp(self):
        dtype = DT
        x_shape = [BS, MS, KS]
        self.x = paddle.randn(x_shape, dtype=dtype)
        self.x.stop_gradient = True

        y_shape = [KS, NS]
        self.y = paddle.randn(y_shape, dtype=dtype)
        self.y.stop_gradient = True

        b_shape = [NS]
        self.b = paddle.randn(b_shape, dtype=dtype)
        self.b.stop_gradient = True

        e1_shape = [BS, MS, NS]
        self.e1 = paddle.randn(e1_shape, dtype=dtype)
        self.e1 = paddle.abs(self.e1) + 1.0
        self.e1.stop_gradient = True

        e2_shape = [BS, MS, NS]
        self.e2 = paddle.randn(e2_shape, dtype=dtype)
        self.e2.stop_gradient = True

        e3_shape = [BS, MS, NS]
        self.e3 = paddle.randn(e3_shape, dtype=dtype)
        self.e3.stop_gradient = True

    def get_subgraph(self):
        B = pct.DimVar(BS)
        M = pct.DimVar(MS)
        K = pct.DimVar(KS)
        N = pct.DimVar(NS)
        T = pct.DTypeVar("T", DT)

        def foo(
            x: pct.Tensor([B, M, K], T),
            y: pct.Tensor([K, N], T),
            b: pct.Tensor([N], T),
            e1: pct.Tensor([B, M, N], T),
            e2: pct.Tensor([B, M, N], T),
            e3: pct.Tensor([B, M, N], T),
        ):
            mm_out = paddle.matmul(x, y)
            add_out = mm_out + b
            div_out = add_out / e1
            mul_out = div_out * e2
            out = mul_out + e3
            return out

        return foo

    def check_if_ap_variadic_exist(self, fused_foo, foo_args):
        generated_pir_program = GetPirProgram(fused_foo, foo_args)
        assert (
            "pd_op.ap_variadic" in generated_pir_program
        ), "AP fusion failed, none pd_op.ap_variadic found in the pir_program."

    def check_by_profiler(self, fused_foo, foo_args):
        paddle.device.synchronize()

        iters = 10
        with profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            on_trace_ready=profiler.export_chrome_tracing("./profiler_log"),
            timer_only=False,
        ) as prof:
            for _ in range(iters):
                _ = fused_foo(*foo_args)
                prof.step()
        prof.summary(
            sorted_by=profiler.SortedKeys.GPUTotal,
            op_detail=True,
            thread_sep=False,
            time_unit="us",
        )

    def test_subgraph(self):
        foo = self.get_subgraph()
        foo_args = (self.x, self.y, self.b, self.e1, self.e2, self.e3)

        iluvatar_gpu_dir = Path(__file__).resolve().parent.parent.parent
        fused_foo = pcc.compile(
            foo,
            ap_path=f"{iluvatar_gpu_dir}/apy/device",
            backend_device="custom_device",
        )

        self.check_if_ap_variadic_exist(fused_foo, foo_args)
        self.check_by_profiler(fused_foo, foo_args)

        ap_outs = fused_foo(*foo_args)
        dy_outs = foo(*foo_args)
        for dy_out, ap_out in zip(dy_outs, ap_outs):
            np.testing.assert_allclose(dy_out, ap_out, rtol=5e-2, atol=1e-1)


if __name__ == "__main__":
    unittest.main()
