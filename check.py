python - <<'PY'
import paddle

print("Paddle version :", paddle.__version__)
print("Compiled CUDA  :", paddle.is_compiled_with_cuda())
print("CUDA version   :", paddle.version.cuda())
print("Device         :", paddle.device.get_device())

x = paddle.randn([2, 3, 320, 320])
y = x * 2
print("CUDA tensor test:", y.shape)
PY
