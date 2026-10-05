# Team responsibilities

| Member | Responsibility | Planning contribution |
|---|---|---|
| Sergey Peshkov | Dataset preparation | Select the dataset and animal categories; inspect annotations; define preprocessing and train/validation/test splits; report subset sizes and leakage checks. |
| Arina Nikolaeva | Model and implementation | Specify Faster R-CNN ResNet-50 FPN and pretrained weights; describe the software stack, GPU resources and implementation pipeline. |
| Anton Kropotov | Experiments and evaluation | Define baseline and comparison, training conditions, metrics, sensitivity experiment and failure-analysis protocol. |
| avlaptev | Project scope and document assembly | Formulate the objective, research question and boundaries; document responsibilities; combine the sections and check consistency. |

Each member reviews their section after integration. Model and experiment owners agree on the class mapping, augmentation settings and evaluation protocol before implementation. The dataset owner supplies the final class list and split statistics after preparation. The document assembly owner records these decisions in the shared plan.

These are planning responsibilities, not claims of completed implementation or experiments. Actual contributions will be updated in the final technical summary. Every member prepares to explain the overall model, metrics, results and failure cases during the individual understanding check.
