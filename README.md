# SparseUNet4D

SparseUNet4D is a 4D sparse-convolutional network for LiDAR moving-object
segmentation (MOS). It combines appearance and temporal-motion evidence in
separate encoders, fuses them in a shared decoder, and applies an object-level
consistency head so points belonging to the same object receive coherent motion
predictions.

The repository includes training and evaluation code for SemanticKITTI, a
streaming Python API, a ROS 2 node, KITTI replay tools, and Husky/Gazebo
deployment instructions.

> This is research software. Do not connect its output directly to a vehicle's
> steering, braking, or emergency-stop controller.

## Architecture

```mermaid
flowchart LR
    A[Five registered LiDAR scans<br/>offsets 0, 1, 2, 4, 8] --> B[4D voxelization<br/>0.1 m]
    B --> C[Appearance encoder<br/>remission]
    B --> D[Motion encoder<br/>temporal residuals]
    C --> E[Bottleneck fusion]
    D --> E
    C -. skip features .-> F[Shared sparse decoder]
    D -. skip features .-> F
    E --> F
    F --> G[Motion head<br/>static / moving]
    F --> H[Semantic head<br/>20 classes]
    F --> I[Center-offset head]
    G --> J[Object-consistency clustering]
    H --> J
    J --> K[Per-point motion labels]
```

For each reference scan, four earlier scans are registered into its coordinate
frame. Signed temporal residuals describe motion while remission supplies
appearance. Two sparse 4D encoders keep these signals separate; their bottleneck
and skip features are fused by a shared decoder. The cluster-consistency head
pools reference-frame foreground voxels and refines their motion logits at the
object level.

## Results

On SemanticKITTI validation sequence 08, the released checkpoint records
**83.66% point-level moving IoU** at its selected threshold of **0.9**. This is a
validation result, not a SemanticKITTI test-server result.

Runtime was measured on 320 consecutive sequence-08 scans (about 123k input
points and 302k five-frame voxels per scan):

| Hardware | Measurement | Latency | Throughput |
|---|---:|---:|---:|
| RTX 3090 | network only, 100 scans | 186.7 ms mean | 5.3 Hz |
| RTX 3090 | end to end, 300 scans after 20 warm-up | 494.1 ms mean | 2.0 Hz |

End to end includes CPU preprocessing, GPU inference, and voxel-to-point
mapping. See [RESULTS.md](RESULTS.md) for experiments and limitations.

## Quick start

1. Follow [INSTALLATION.md](INSTALLATION.md) to install the verified Python,
   PyTorch, CUDA, and MinkowskiEngine environment.
2. Download the released checkpoint:

   ```bash
   bash deploy/download_model.sh
   ```

3. Follow [DEPLOYMENT.md](DEPLOYMENT.md) for SemanticKITTI replay, ROS 2,
   Gazebo, and real-robot commands.

The checkpoint is installed at:

```text
checkpoints/sparseunet4d_semantickitti/best.pt
```

It is distributed as a GitHub Release asset because its 141 MB size exceeds
GitHub's normal per-file repository limit. Do not commit it directly to Git.

## Repository layout

```text
configs/                    model, training, and deployment configurations
deploy/                     download, replay, benchmark, and setup utilities
scripts/                    training and robustness scripts
sparseunet4d/datasets/      SemanticKITTI loading and temporal features
sparseunet4d/models/        sparse backbone, dual branch, heads, and losses
mos_inference.py            ROS-independent streaming inference API
mos_node.py                 ROS 2 streaming node
INSTALLATION.md             beginner installation guide
DEPLOYMENT.md               running and deployment guide
README_HUSKY_GAZEBO.md      isolated Docker deployment on Husky/Gazebo
```

## Training and evaluation

After setting SemanticKITTI paths in the selected YAML file:

```bash
export SU4D_BACKEND=me
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

python scripts/train.py \
  --config configs/pretrained_semantickitti.yaml \
  --save-dir runs/my_experiment
```

For official point-level validation, provide the MOS label mapping required by
the evaluator:

```bash
python eval_mos_official.py \
  --config configs/pretrained_semantickitti.yaml \
  --ckpt checkpoints/sparseunet4d_semantickitti/best.pt \
  --mos-yaml /path/to/mos-label-mapping.yaml \
  --point-level \
  --threshold 0.9
```

## Citation and license

The paper citation and software license have not yet been added. Add both before
an archival public release so users know how to cite the work and what reuse is
permitted.
