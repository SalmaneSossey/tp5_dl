# Knowledge Distillation Report

## Objective

This submission implements the ENSIAS Deep Learning TP on knowledge distillation in two parts. Part 1 studies response-based KD and Attention Transfer on a filtered MNIST task with digits `0`, `1`, and `8`. Part 2 studies classical KD, FitNets, and RKD on a filtered CIFAR-10 task with the classes `cat`, `dog`, `deer`, and `horse`.

## Part 1 Methods

- Dataset: MNIST filtered to `0`, `1`, and `8`, remapped to labels `0`, `1`, and `2`, resized to `32x32`, normalized with mean `0.5` and std `0.5`.
- Teacher: pretrained `ResNet-50` adapted to `3` outputs.
- Main student: `MicroCNN` with three `Conv -> BN -> ReLU` blocks and a final fully connected layer.
- Additional studies: temperature sweep, three student-size regimes, and Attention Transfer.
- Metrics: parameter count, approximate model size, latency, precision, and test accuracy.

## Part 1 Main Results

### Teacher and Student Statistics

| model              | params   | params_k  | approx_size_kb | approx_size |
| ------------------ | -------- | --------- | -------------- | ----------- |
| Teacher - ResNet50 | 23514179 | 23514.179 | 92060.175781   | 89.90 MB    |
| Student - MicroCNN | 21075    | 21.075    | 83.097656      | 83.10 KB    |

### MNIST Teacher Fine-Tuning History

| epoch | train_loss | train_acc | test_loss | test_acc |
| ----- | ---------- | --------- | --------- | -------- |
| 1     | 0.198601   | 0.935557  | 0.009508  | 0.998389 |
| 2     | 0.022684   | 0.993361  | 0.012677  | 0.996456 |
| 3     | 0.010687   | 0.99687   | 0.009741  | 0.997745 |
| 4     | 0.010395   | 0.997917  | 0.005946  | 0.998389 |
| 5     | 0.010571   | 0.997032  | 0.004998  | 0.998389 |

### Experiment 1 Summary

| model         | size     | latency_ms | precision | IoT chip OK? |
| ------------- | -------- | ---------- | --------- | ------------ |
| Student alone | 83.10 KB | 0.388224   | 0.99839   | Yes          |
| Student + KD  | 83.10 KB | 0.365703   | 0.99966   | Yes          |

### Temperature Sweep

| temperature | test_acc |
| ----------- | -------- |
| 1.0         | 0.999034 |
| 2.0         | 0.998067 |
| 4.0         | 0.999034 |
| 8.0         | 0.998067 |

### Distillation Regimes

| student        | params   | accuracy without KD | accuracy with KD |
| -------------- | -------- | ------------------- | ---------------- |
| Large Student  | 11178051 | 0.999034            | 0.999034         |
| Medium Student | 21075    | 0.998711            | 0.999356         |
| Small Student  | 3283     | 0.997745            | 0.997101         |

### Final Part 1 Synthesis

| model             | size     | latency_ms | precision | IoT chip OK? |
| ----------------- | -------- | ---------- | --------- | ------------ |
| Teacher           | 89.90 MB | 6.223351   | 0.998294  | No           |
| Student alone     | 83.10 KB | 0.388224   | 0.99839   | Yes          |
| Student + KD      | 83.10 KB | 0.365703   | 0.99966   | Yes          |
| Student + KD + AT | 83.10 KB | 0.363814   | 0.999024  | Yes          |

Part 1 is fully executed. The compact student clearly dominates the teacher for deployment. In the observed runs, `Student + KD` achieved the best measured precision (`0.999660`) while keeping the same tiny footprint (`83.10 KB`) and very low latency (`0.365703 ms`). The teacher remained far too large (`89.90 MB`) for an IoT target.

## Part 2 Methods

- Dataset: CIFAR-10 filtered to `cat`, `dog`, `deer`, and `horse`, remapped to labels `0..3`.
- Teacher: pretrained `VGG-16` adapted to `4` outputs.
- Student: `TinyCNN` with convolution widths `16 -> 32 -> 64 -> 128`, batch normalization, ReLU, max-pooling after layers `2` and `4`, and a final fully connected layer.
- Distillation methods: baseline supervised training, classical KD, FitNets, and RKD.
- Completed outputs: teacher training history, baseline reference, FitNets comparison, and RKD comparison.

## Part 2 Available Results

### Teacher and Student Statistics

| model             | params    | params_k   | approx_size_kb | approx_size |
| ----------------- | --------- | ---------- | -------------- | ----------- |
| Teacher - VGG16   | 134276932 | 134276.932 | 524519.265625  | 512.23 MB   |
| Student - TinyCNN | 98196     | 98.196     | 385.484375     | 385.48 KB   |

### CIFAR Teacher Fine-Tuning History

| epoch | train_loss | train_acc | test_loss | test_acc |
| ----- | ---------- | --------- | --------- | -------- |
| 1     | 0.61011    | 0.767821  | 0.418892  | 0.840278 |
| 2     | 0.373737   | 0.858726  | 0.41658   | 0.846478 |
| 3     | 0.274399   | 0.901458  | 0.370471  | 0.864087 |
| 4     | 0.217447   | 0.920327  | 0.371704  | 0.867063 |
| 5     | 0.162306   | 0.943241  | 0.409364  | 0.872024 |

### Baseline Reference

| model         | test_acc |
| ------------- | -------- |
| Student alone | 0.762153 |
| Student + KD  | 0.736359 |

### FitNets Results

| gamma | accuracy |
| ----- | -------- |
| 0.1   | 0.707589 |
| 0.5   | 0.80754  |
| 1.0   | 0.721974 |

### RKD Results

| configuration | lambda_D | lambda_A | accuracy |
| ------------- | -------- | -------- | -------- |
| RKD-D only    | 1.0      | 0.0      | 0.772321 |
| RKD-A only    | 0.0      | 2.0      | 0.752728 |
| RKD combined  | 1.0      | 2.0      | 0.744792 |

Among the completed Part 2 runs, the strongest student result came from **FitNets with `gamma = 0.5`**, which reached `0.807540` accuracy. This was better than the supervised baseline student (`0.762153`), better than classical KD (`0.736359`), and better than all tested RKD variants (`0.772321` best for RKD-D only).

## Pending Items

The final three notebook sections were not executed because the Colab session disconnected before completion and the remaining analyses were too expensive to rerun in the available budget:

- `P2.6` t-SNE on 200 test images
- `P2.7` inter-class similarity matrices
- `P2.8/P2.9` final latency synthesis and the final comparative analysis for the camera deployment scenario

No numerical values have been invented for these pending sections. The repository records this explicitly in `results/pending_items.md`.

## Key Figures Available in the Repository

- `figures/part1_soft_labels_ambiguous_8.png`
- `figures/part1_experiment1_accuracy_curves.png`
- `figures/part1_temperature_effect.png`
- `figures/part1_attention_maps_before_after.png`
- `figures/part2_fitnets_feature_maps.png`

## Deployment Conclusions

- **IoT chip (Part 1):** `MicroCNN` is the clear deployment target. `Student + KD` gives the best measured trade-off between precision, size, and latency.
- **Camera (Part 2):** a rigorous final recommendation cannot be made yet because the final latency and representation-structure analyses were not executed. On the completed accuracy results alone, `FitNets (gamma = 0.5)` is the strongest current student candidate, but this remains provisional.
