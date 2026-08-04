"""
check_env_info.py

Run this in the SAME environment you used to export/simplify/test the
PicoDet ONNX model. It prints out everything you need for the
"Setup Notes" file you send to the client:

  - Opset version baked into the ONNX model
  - IR version and producer (paddle2onnx) info
  - Installed onnx / onnxruntime / onnxsim / paddle2onnx versions
  - Model input names, shapes, and types

Usage:
    python check_env_info.py path/to/picodet_standard_sim.onnx
"""

import sys
import importlib.metadata as metadata


def get_version(pkg_name):
    try:
        return metadata.version(pkg_name)
    except metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_env_info.py path/to/model.onnx")
        sys.exit(1)

    model_path = sys.argv[1]

    import onnx
    import onnxruntime as ort

    model = onnx.load(model_path)

    print("=" * 55)
    print("ONNX MODEL INFO")
    print("=" * 55)
    print(f"File:            {model_path}")
    print(f"Opset version:   {model.opset_import[0].version}")
    print(f"IR version:      {model.ir_version}")
    print(f"Producer:        {model.producer_name} {model.producer_version}")

    print()
    print("=" * 55)
    print("INSTALLED PACKAGE VERSIONS")
    print("=" * 55)
    print(f"onnx:            {get_version('onnx')}")
    print(f"onnxruntime:     {get_version('onnxruntime')}")
    print(f"onnxsim:         {get_version('onnxsim')}")
    print(f"paddle2onnx:     {get_version('paddle2onnx')}")
    print(f"paddlepaddle:    {get_version('paddlepaddle')}")
    print(f"numpy:           {get_version('numpy')}")

    print()
    print("=" * 55)
    print("MODEL INPUT / OUTPUT INFO (via onnxruntime)")
    print("=" * 55)
    sess = ort.InferenceSession(model_path)

    print("Inputs:")
    for i in sess.get_inputs():
        print(f"  - name: {i.name:20s} shape: {i.shape}  type: {i.type}")

    print("Outputs:")
    for o in sess.get_outputs():
        print(f"  - name: {o.name:20s} shape: {o.shape}  type: {o.type}")

    print()
    print("Copy the two sections above directly into your setup notes.")


if __name__ == "__main__":
    main()
