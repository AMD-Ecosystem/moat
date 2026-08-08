# DynOSAM notes

## Why this is back

Re-opened 2026-08-07 after two premises changed.

The prior screen declined this on TensorRT and OpenCV CUDA. Both have answers now. Its TensorRT use is ONNX load plus inference in a single file (`dynosam_nn/src/YoloV8ObjectDetector.cc`), which is the shape MIGraphX is built for -- rewritten integration, not a rewritten project. And `cv::cuda::*` is covered by MOAT's own opencv_contrib port, completed on four architectures and upstream at opencv/opencv_contrib#4147, so this wants `depends_on: [opencv_contrib]` rather than a decline.

Check before assuming either: whether it reaches TensorRT through ONNX Runtime, in which case the MIGraphX execution provider may need no application change at all; and which `cv::cuda` calls it actually makes against what opencv_contrib ported.

## The prior analysis

Do not redo it. The earlier screen's full write-up is in history:

    git show b40576d53399:projects/DynOSAM/plan.md
    git show b40576d53399:projects/DynOSAM/notes.md

Read it first and test only what has changed.
