# Blur-Aware Self-Supervised Pretraining for Object Detection in Autonomous Driving

Motion blur from rapid motion or adverse weather de
grades object detection performance in autonomous driving.
We propose a blur-aware self-supervised pretraining strat
egy that uses synthetic blur during contrastive learning to
produce encoders more robust to blurred inputs. We pretrain
a ResNet-18 encoder with SimCLR on synthetically blurred
KITTI images and integrate it into DETR for downstream
object detection. Experiments show substantial reductions in
bounding-box losses and faster convergence, though Average
Precision (AP) did not improve under the current training
setup. We analyze possible causes and outline practical
improvement directions.

# Method

# 3.1. Overview
Our pipeline has two stages: (1) self-supervised pretrain
ing of a ResNet-18 encoder using SimCLR with heavy usage
of synthetic blur augmentations, and (2) integration of the
pretrained encoder into a DETR detection head followed by
supervised fine-tuning on labeled KITTI images (clean and
mixed clean+blur).
# 3.2. Dataset and Synthetic Blur
WeusetheKITTI2D objectdetection benchmark [Geiger
et al., 2012] (7k images with annotations for car, pedestrian,
and cyclist). Since KITTI lacks motion blur, we synthesize
Gaussian blur at three intensities (low/medium/high) applied
as an augmentation during SSL pretraining and optionally
during fine-tuning.
# 3.3. Self-Supervised Pretraining
We adopt SimCLR with ResNet-18 as the encoder back
bone. Key augmentations: random crop, color jitter, and
synthetic Gaussian blur (random intensity). Pretraining ob
jective is the NT-Xent contrastive loss. The encoder’s pro
jection head is discarded after pretraining and the backbone
weights are transferred to DETR.
# 3.4. Fine-tuning: DETR
We replace DETR’s [Carion et al., 2020] CNN backbone
with the pretrained ResNet-18 and fine-tune the entire DETR
model on supervised KITTI training splits. We compare
three baselines:
1. DETRwith ResNet-50 pretrained on COCO [Lin et al.,
2015]: standard baseline
2. DETR+ResNet-18: fine-tuned on KITTI Dataset
3. DETR +ResNet-18: ResNet-18 pretrained with Sim
• Limited diversity of synthetic blur types — Gaussian
CLRonsynthetically blurred KITTI and the final model
fine-tuned on KITTI Dataset (ours)

# 4.2. Quantitative Results
Table 1 summarizes representative metrics. The blur
pretrained encoder achieved large relative reductions in
bounding-box losses across multiple DETR layers, but AP
scores remained low under the current training regimen.


<img width="523" height="163" alt="image" src="https://github.com/user-attachments/assets/2ba3329b-a681-455a-8f62-a3e2a7abf8df" />



# 4.3. Analysis
It is important to acknowledge, that there are no pre
trained light models available, that is why all of the ResNet
18 models are trained from scratch.
The observed large reductions in bbox losses indicate that
blur-aware SSL helps the encoder learn more stable low
and mid-level features that lead to better regression behavior.
However, AP not improving (even 0.0 in our runs) suggests
that detection heads did not sufficiently adapt during fine
tuning. Possible reasons:
• Insufficient no. epochs for the detection head to learn
class discrimination on top of the SSL features.
• A domain gap between SSL objectives (instance
discrimination) and detection objectives (localiza
tion+classification).
• Hyperparameters (learning rate schedule, weight de
cay, DETR-specific losses) not tuned for transferred
encoders.



