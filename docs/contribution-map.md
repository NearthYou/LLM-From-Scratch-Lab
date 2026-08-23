# 구현 역할

원본 과제는 4명이 페어 프로그래밍으로 진행했습니다. 이시원은 다음 영역을 구현하고 두 과제의 실행 흐름을 하나의 저장소로 다시 구성했습니다.

## GPT lab

원본: [`Soldbone/gpt-lab`](https://github.com/Soldbone/gpt-lab)

| 영역 | 현재 위치 | 구현 내용 |
| --- | --- | --- |
| Loss | `gpt/src/train.py` | single batch loss, DataLoader 평균 loss |
| Checkpoint | `gpt/src/train.py` | model과 optimizer 저장, epoch와 step 복원 |
| Generation | `gpt/src/train.py` | temperature, top-k, encode에서 decode까지의 sample 흐름 |
| Pretraining | `gpt/src/train.py` | train loop, evaluation과 checkpoint 연결 |
| Fine-tuning | `gpt/src/finetune.py` | sentiment dataset, classifier, train과 evaluate 도구 |
| Regression | `gpt/tests/` | generation과 training 경계 test |
| Experiment | `artifacts/pretraining/` | batch-size 설정, epoch metric, summary와 graph |

## NumPy와 MNIST lab

원본: [`devhyun05/group4-mnist-lab`](https://github.com/devhyun05/group4-mnist-lab)

| 영역 | 현재 위치 | 구현 내용 |
| --- | --- | --- |
| Activation | `foundations/mnist_numpy/src/activations.py` | ReLU, Softmax forward와 backward |
| Layer | `foundations/mnist_numpy/src/layers.py` | Affine, Dropout, BatchNorm |
| Loss | `foundations/mnist_numpy/src/losses.py` | mean cross-entropy |
| Optimizer | `foundations/mnist_numpy/src/optimizers.py` | SGD, Adam과 bias correction |
| Network | `foundations/mnist_numpy/src/network.py` | 784→512→256→10 network composition |
| Training | `foundations/mnist_numpy/src/training.py` | mini-batch training, evaluation과 loss plot |

## 검증 명령

```bash
uv run pytest foundations/mnist_numpy/tests gpt/tests -q
uv run pytest tests/test_smoke_train.py tests/test_evidence_contract.py -q
```
