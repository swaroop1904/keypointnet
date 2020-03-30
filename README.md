# keypointnet
This is an reimplementation of the keypoint network proposed in "Discovery of
Latent 3D Keypoints via End-to-end Geometric Reasoning
[[pdf](https://arxiv.org/pdf/1807.03146.pdf)]". The keypointnet predicts a consistent set of keypoint given a single image. The predicted keypoint can then be used for various downstream tasks such as detection and pose estimation.  


```
@inproceedings{suwajanakorn2018discovery,
  title={Discovery of latent 3d keypoints via end-to-end geometric reasoning},
  author={Suwajanakorn, Supasorn and Snavely, Noah and Tompson, Jonathan J and Norouzi, Mohammad},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2059--2070},
  year={2018}
}

```

The functions defined in this repo has been either adapted from or directly taken (Transformer class, blender render script and few of the loss function) from https://github.com/tensorflow/models/tree/master/research/keypointnet and follows the original license under the original repo. 