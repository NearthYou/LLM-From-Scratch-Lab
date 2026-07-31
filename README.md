# LLM From Scratch Lab

NumPy로 만든 MNIST 신경망에서 출발해 BPE, embedding, causal multi-head attention, GPT 사전학습과 생성, 감성 분류 미세조정 유틸리티까지 이어지는 학습 결과를 한 저장소에서 검증할 수 있도록 재구성했습니다.

원본은 두 개의 팀 과제 저장소에 나뉘어 있습니다.

- GPT 원본: [`Soldbone/gpt-lab`](https://github.com/Soldbone/gpt-lab)
- NumPy/MNIST 원본: [`devhyun05/group4-mnist-lab`](https://github.com/devhyun05/group4-mnist-lab)

재배포 동의와 라이선스 경계는 [ATTRIBUTION.md](ATTRIBUTION.md)에 기록했습니다

## 프로젝트 요약

| 항목 | 내용 |
| --- | --- |
| 개발 형태 | 4인 페어 프로그래밍 팀 과제, 이후 개인 검증 저장소로 재구성 |
| 확인하려던 것 | LLM API를 사용하는 수준을 넘어 tokenization부터 attention, 학습과 생성까지 데이터가 어떻게 흐르는지 직접 확인 |
| 구현 범위 | byte-level BPE, embedding, causal multi-head attention, Transformer block, 학습과 생성 |
| 실험 | 같은 seed, 10 epochs와 150만 character corpus에서 batch size만 4, 8, 16으로 변경 |
| 검증 원칙 | 당시 GPU 실험과 현재 CPU smoke run을 분리하고 개인, 페어, 팀 기여를 commit 근거로 구분 |

## 학습 흐름

```text
NumPy tensors
  -> Affine / BatchNorm / ReLU / Dropout
  -> Softmax + Cross Entropy
  -> SGD / Adam
  -> MNIST classifier

text
  -> byte-level BPE
  -> token ids + positional ids
  -> embeddings
  -> causal multi-head attention
  -> residual Transformer blocks
  -> language-model head
  -> next-token loss / generation
  -> sentiment classifier fine-tuning utilities
```

자세한 데이터 흐름과 모듈 경계는 [docs/architecture.md](docs/architecture.md)에 있습니다.

## 구현 범위

| 영역 | 구현과 검증 범위 | 위치 |
| --- | --- | --- |
| NumPy 기초 | Affine, ReLU, Softmax, BatchNorm, Dropout, cross entropy, SGD, Adam, 학습 루프 | `foundations/mnist_numpy/` |
| Tokenization | byte-level BPE 학습, encode/decode, vocabulary와 merge rule | `gpt/src/bpe.py` |
| GPT | token/position embedding, causal multi-head attention, LayerNorm, FFN, residual blocks, LM head | `gpt/src/` |
| Training | loss 계산, checkpoint, 생성, pretraining loop | `gpt/src/train.py` |
| Fine-tuning | 감성 데이터셋, 분류 헤드, 학습 및 평가 유틸리티 | `gpt/src/finetune.py` |
| 증빙 | 현재 CPU smoke JSON, 과거 GPU run의 선택 아티팩트, 기여 commit map | `artifacts/`, `docs/` |

## 빠른 검증

Python 3.11과 [`uv`](https://docs.astral.sh/uv/)를 기준으로 합니다.

```bash
uv sync
uv run pytest -q
uv run python scripts/smoke_train.py --output artifacts/current/smoke-result.json
```

전체 suite는 기능 테스트 50개, CPU smoke 테스트 1개, 증빙 계약 테스트 3개로 구성됩니다.

## Historical result - commit `4fe533e`

아래 수치는 2026-06-03에 CUDA 환경에서 실행해 원본 commit `4fe533e`에 저장한 과거 batch-size 비교 결과입니다. 현재 CPU smoke 결과가 아닙니다.

- seed 42, 10 epochs, corpus 1.5M characters
- vocabulary 3,000, context 128, embedding 128
- 2 Transformer layers, 4 heads, dropout 0.1
- 현재 코드로 계산한 모델 parameter 수: 1,180,416

| Batch | Best validation loss | Test perplexity | Test Top-1 / Top-3 / Top-5 | Total time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 4.0773 | 59.3438 | 0.2272 / 0.3745 / 0.4476 | 714.12 s |
| 8 | 4.1646 | 64.9118 | 0.2178 / 0.3618 / 0.4344 | 451.79 s |
| 16 | 4.2240 | 68.9680 | 0.2131 / 0.3555 / 0.4266 | 312.58 s |

이 한 번의 실험에서는 batch 16이 batch 4보다 전체 학습 시간이 약 56% 짧았지만, best validation loss와 test perplexity는 batch 4가 더 좋았습니다. 큰 batch가 항상 더 좋은 결과를 낸다고 일반화하지 않고, 같은 데이터와 설정에서 확인한 학습 시간과 품질의 trade-off로만 해석합니다.

![Historical validation-loss comparison](artifacts/pretraining/loss_comparison_val.png)

원본 config, epoch metrics, 표, dashboard는 [`artifacts/pretraining/`](artifacts/pretraining/)에 보존했습니다. 해석과 출처 경계는 [docs/results.md](docs/results.md)를 함께 확인해 주세요.

## Current CPU smoke result - 2026-07-18

현재 환경에서 고정 synthetic batch를 사용해 작은 GPT를 CPU로 5 step 학습한 결과입니다.

| Device | Seed | Steps | Initial loss | Final loss | Torch |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 42 | 5 | 4.308793 | 2.670516 | 2.13.0+cpu |

이 smoke run은 설치, forward/backward, optimizer update가 함께 동작하고 같은 batch에서 loss가 감소하는지만 확인합니다. 과거 10-epoch GPU 실험의 성능을 재현했다는 뜻은 아닙니다. 원본 JSON은 [`artifacts/current/smoke-result.json`](artifacts/current/smoke-result.json)입니다.

## 프로젝트 구조

```text
.
├── foundations/mnist_numpy/   # NumPy neural-network foundation
├── gpt/                       # BPE, GPT, training and fine-tuning code
├── scripts/smoke_train.py     # deterministic CPU smoke run
├── artifacts/current/         # current-environment result
├── artifacts/pretraining/     # selected files from commit 4fe533e
├── docs/architecture.md       # model and data flow
├── docs/results.md            # historical/current result boundary
├── docs/contribution-map.md   # personal commit evidence
└── ATTRIBUTION.md             # source and reuse status
```

## 기여와 출처

- 팀 구현 전체의 출처와 재사용 조건: [ATTRIBUTION.md](ATTRIBUTION.md)
- 이시원(`NearthYou`)의 원본 commit별 기여: [docs/contribution-map.md](docs/contribution-map.md)
- GPT 원본 개선 PR: [`Soldbone/gpt-lab#40`](https://github.com/Soldbone/gpt-lab/pull/40)
- fork의 개인 증빙 PR: [`NearthYou/gpt-lab#1`](https://github.com/NearthYou/gpt-lab/pull/1)

## 한계

- 현재 재현은 synthetic batch를 사용하는 CPU smoke check입니다. 모델 품질 평가나 corpus-level benchmark가 아닙니다.
- 보존된 sample-generation 표에는 당시 데이터 인코딩 문제로 깨진 문자가 포함되어 있어 품질 근거로 사용하지 않습니다.
- 감성 분류 fine-tuning 코드와 테스트는 있지만, 검증 가능한 historical accuracy artifact가 없어 정확도 수치를 주장하지 않습니다.
