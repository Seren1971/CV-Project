# Implementation plan

*Owner: Arina Nikolaeva. Consolidated from the implementation outline supplied by the team.*

## Model and environment

Use Faster R-CNN with a ResNet-50 FPN backbone and COCO-pretrained weights from torchvision. The software stack is Python, PyTorch, torchvision, Pillow and NumPy. Training and evaluation run on a GPU in Google Colab or Kaggle.

## Pipeline

1. Download CCT20 images and annotations and prepare the agreed splits.
2. Load images and bounding boxes with the custom PyTorch dataset.
3. Adapt the detector output to the selected CCT20 categories and fine-tune it on the training split.
4. Run the baseline and augmentation variant according to the experimental plan.
5. Select a checkpoint and confidence threshold using validation data.
6. Evaluate the selected models on test data and save metrics, predictions and failure examples.

The exact torchvision model constructor, weight identifier, package versions and input transforms will be recorded by the implementation owner before execution. Images and boxes must be transformed together during spatial augmentation. The unchanged COCO reference uses a separate class-agnostic evaluation mapping rather than the newly initialised CCT20 prediction head.
