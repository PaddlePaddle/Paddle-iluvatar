export FLAGS_prim_all=True
export FLAGS_prim_enable_dynamic=true
export FLAGS_use_cinn=1

rm -rf /tmp/paddle/ap_workspace/*

python test_ap_matmul_add_relu_iluvatar.py
# python test_ap_matmul_add_multiply_iluvatar.py
# python test_ap_matmul_add_gelu_iluvatar.py
# python test_ap_matmul_add_divide_multiply_iluvatar.py
# python test_ap_matmul_add_divide_multiply_add_iluvatar.py
