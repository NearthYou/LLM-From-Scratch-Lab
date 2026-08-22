# 구조와 학습 경로

이 저장소는 신경망의 기본 연산을 직접 구현하는 단계와 PyTorch로 자기회귀 언어 모델을 조립하는 단계를 분리합니다.

## 전체 데이터 흐름

```text
text -> byte-level BPE -> token ids
token ids + positions -> embedding
embedding -> causal multi-head attention -> residual/FFN blocks
hidden states -> LM head -> next-token loss
checkpoint -> generation or sentiment classifier fine-tuning
```

## 1. NumPy와 MNIST 기초

`foundations/mnist_numpy/`는 자동미분 없이 forward cache와 backward gradient를 직접 연결합니다.

```text
(batch, 784)
  -> Affine(784, 512)
  -> BatchNorm -> ReLU -> Dropout
  -> Affine(512, 256)
  -> BatchNorm -> ReLU -> Dropout
  -> Affine(256, 10)
  -> Softmax -> Cross Entropy
```

| Module | Responsibility |
| --- | --- |
| `activations.py` | ReLU mask와 안정적인 row-wise Softmax |
| `layers.py` | Affine, BatchNorm, Dropout의 forward/backward |
| `losses.py` | integer label 기반 mean cross entropy |
| `network.py` | ordered layer composition과 gradient 수집 |
| `optimizers.py` | SGD와 bias-corrected Adam update |
| `training.py` | mini-batch shuffle, loss, backpropagation, evaluation |

이 단계에서는 최종 accuracy보다 tensor shape, cached state, gradient 흐름과 train/eval mode의 차이를 코드와 단위 테스트로 확인합니다.

## 2. Byte-level BPE

`gpt/src/bpe.py`는 text를 byte sequence로 바꾸고 빈도가 높은 adjacent pair를 merge해 vocabulary를 구성합니다. 학습된 `vocabulary.json`과 `merge_rule.json`은 encode/decode의 재현 입력입니다.

```text
UTF-8 text
  -> bytes
  -> special-token-aware initial ids
  -> learned pair merges
  -> token ids
```

## 3. Embedding과 causal attention

`InputEmbedding`은 token embedding과 position embedding을 더한 뒤 dropout을 적용합니다. `MultiHeadAttention`은 같은 hidden state에서 Q/K/V projection을 만들고, future token을 보지 못하도록 upper-triangular causal mask를 적용합니다.

```text
token ids: (batch, sequence)
  -> embeddings: (batch, sequence, d_model)
  -> split heads: (batch, heads, sequence, head_dim)
  -> scaled QKᵀ + causal mask
  -> attention weights @ V
  -> concatenate heads + output projection
```

## 4. GPT block과 language model head

`gpt/src/model.py`의 `TransformerBlock`은 pre-normalization과 두 residual path를 사용합니다.

```text
x -> LayerNorm -> causal MHA -> Dropout -> +x
  -> LayerNorm -> Linear/GELU/Linear -> Dropout -> +x
```

여러 block의 hidden state는 final LayerNorm과 vocabulary projection을 지나 token별 logits가 됩니다. target이 주어지면 모든 position을 flatten해 next-token cross entropy를 계산합니다.

## 5. 학습, checkpoint, 생성과 fine-tuning

| Path | Boundary |
| --- | --- |
| `gpt/src/train.py` | batch/loader loss, optimizer step, evaluation, checkpoint save/load, token generation |
| `gpt/src/finetune.py` | sentiment dataset, classifier head, fine-tuning/evaluation utility |
| `scripts/smoke_train.py` | 고정 synthetic batch에서 CPU forward/backward/update 연결 확인 |

Generation은 context window를 잘라 model logits을 얻은 뒤 greedy 또는 configured sampling으로 다음 token을 붙입니다. Fine-tuning utility는 pretraining checkpoint를 classifier에 연결할 수 있지만, 이 저장소에는 검증 가능한 sentiment accuracy artifact가 없어 성능 수치를 별도로 주장하지 않습니다.

## 검증 경계

- `foundations/mnist_numpy/tests/`: NumPy layer, optimizer와 network 동작
- `gpt/tests/`: BPE, embedding/attention, GPT, training, fine-tuning 동작
- `tests/test_smoke_train.py`: 현재 CPU 환경의 optimizer-connected smoke run
- `tests/test_evidence_contract.py`: 출처, historical/current 구분, active assignment stub 부재

과거 GPU 실험과 현재 CPU smoke run은 설정, 데이터와 목적이 다르므로 직접 비교하지 않습니다.
