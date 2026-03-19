# PaddlePaddle for Iluvatar GPU

English | [简体中文](./README_cn.md)

请参考以下步骤进行编译、安装和验证 paddlepaddle_iluvatar。

## 编译安装

```bash
# 请联系天数智芯客户支持 (services@iluvatar.com) 获取 SDK 镜像

# 克隆 PaddleCustomDevice 源代码
git clone https://github.com/PaddlePaddle/Paddle-iluvatar.git

bash build_paddle.sh

# 安装
bash install_paddle.sh
```

## 增量编译（代码修改后更快地重新编译）
```bash
# 增量编译（代码修改后更快地重新编译，也会安装 whl 包）
bash build_inc.sh
```

## 验证

```bash
# 运行测试
cd tests
bash run_test.sh
```
