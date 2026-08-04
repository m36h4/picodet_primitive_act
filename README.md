PicoDet ONNX - v2 delivery (2026-08-04)

This replaces the version shared earlier. The previous version failed with:
[ONNXRuntimeError] Node (p2o.Gather.8) Op (Gather) [ShapeInferenceError] data tensor must have rank >= 1

Fixed by re-running the export through onnx-simplifier. Tested on two
separate machines (Windows CPU, Linux GPU over SSH) and it loads/runs on
both.

Files:
  models/standard_picodet_sim.onnx   - standard activations, opset 12
  models/primitive_picodet_sim.onnx  - hard-sigmoid/hard-swish, opset 13
    (use this one only if your hardware doesn't support standard
    sigmoid/swish ops)

Dependencies (see requirements.txt):
  onnxruntime==1.23.2
  numpy==1.26.4

You do NOT need paddle, paddle2onnx, or onnxsim to run these models.
Those were only used on our end to export/simplify them.

Setup:
  python -m venv onnx_infer_env
  source onnx_infer_env/bin/activate      (Windows: onnx_infer_env\Scripts\activate)
  pip install -r requirements.txt

Check the model loads and matches expected info:
  python scripts/check_env_info.py models/standard_picodet_sim.onnx

Run a dummy forward pass to confirm no load/shape errors:
  python scripts/run_inference_test.py models/standard_picodet_sim.onnx

Model inputs:
  image          [N, 3, 320, 320]  float32
  scale_factor   [1, 2]            float32

Model outputs:
  save_infer_model/scale_0.tmp_0   [N, 6]  float32  (class, score, x1, y1, x2, y2)
  save_infer_model/scale_1.tmp_0   [1]     int32    (box count)

If either script fails on your side, send back the full console output
plus `pip freeze` so we can compare against what worked here.
