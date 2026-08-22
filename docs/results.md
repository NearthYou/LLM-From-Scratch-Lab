# 실험 결과와 검증 범위

이 문서는 발표 자료에 기록된 비교, 원본 저장소에서 가져온 batch 산출물, 현재 환경에서 다시 실행하는 CPU smoke를 구분합니다. 설정과 근거가 다른 결과를 하나의 연속 실험처럼 합치지 않습니다.

## 결과를 읽는 기준

| 구분 | 근거 | 다시 확인할 수 있는 범위 |
| --- | --- | --- |
| 발표 기록 | 독립 저장소 README commit과 보존 이미지 | layer, learning rate, dropout, 최종 설정의 표와 그래프 |
| 과거 원본 산출물 | 원본 `Soldbone/gpt-lab` commit `4fe533e` | batch 4, 8, 16의 설정, epoch 지표, summary와 그래프 |
| 현재 재현 | `artifacts/current/smoke-result.json`과 현재 테스트 | CPU forward, loss, backward, AdamW update 연결 |

발표 기록에는 원본 지표 파일이 남아 있지 않습니다. 표와 이미지는 당시 결과를 설명하는 자료로 보존하지만 현재 환경에서 같은 값이 나온다고 주장하지 않습니다.

## 발표 당시 기준 실행

| 항목 | 값 |
| --- | ---: |
| vocabulary | 3,000 |
| context length | 128 |
| embedding dimension | 128 |
| Transformer layers | 2 |
| attention heads | 4 |
| learning rate | 3e-4 |
| batch size | 8 |
| dropout | 0.1 |
| epochs | 10 |

이 실행은 마지막 epoch에서 train loss 5.278, validation loss 5.565, test loss 5.563을 기록했습니다. 독립 저장소 README에 보존된 발표 기록이며 이 실행의 원본 지표 파일은 현재 저장소에 없습니다.

## 설정 비교

### Transformer 층 수

| Layers | 마지막 validation loss |
| ---: | ---: |
| 2 | 4.160 |
| 4 | 4.067 |
| 6 | 3.995 |

시도한 범위에서는 6층의 validation loss가 가장 낮았습니다. 다른 모델 크기와 말뭉치에서도 층이 많을수록 낫다는 결론으로 확장하지 않습니다.

![Transformer 층 수에 따른 validation loss](images/layer-comparison.svg)

### 학습률

| Learning rate | 마지막 validation loss |
| ---: | ---: |
| 1e-4 | 4.4031 |
| 3e-4 | 4.1608 |
| 5e-4 | 4.0297 |

시도한 범위에서는 5e-4가 가장 낮았습니다. 더 큰 learning rate에서 같은 경향이 이어지는지는 확인하지 않았습니다.

![학습률에 따른 validation loss](images/learning-rate-comparison.png)

### Dropout

| Dropout | Best validation loss |
| ---: | ---: |
| 0.1 | 5.699 |
| 0.2 | 5.793 |
| 0.3 | 5.900 |

이 기록에서는 dropout을 높일수록 train과 validation loss가 함께 올라 0.1을 유지했습니다. 원본 지표 파일이 없어 그래프와 표를 발표 기록으로만 사용합니다.

![dropout에 따른 best validation loss](images/dropout-comparison.svg)

### 배치 크기

batch 비교는 다른 세 비교와 달리 원본 산출물을 보존했습니다. 근거 원본은 [`Soldbone/gpt-lab` commit `4fe533e`](https://github.com/Soldbone/gpt-lab/commit/4fe533e)와 해당 commit의 `outputs/batch_size_experiment_20260603_170551`입니다.

| Field | Value |
| --- | --- |
| Run date | 2026-06-03 |
| Seed | 42 |
| Device | CUDA |
| Epochs | 10 |
| Corpus size | 1,500,000 characters |
| Vocabulary, context | 3,000, 128 |
| Embedding, layers, heads | 128, 2, 4 |
| Dropout | 0.1 |
| Learning rate, weight decay | 0.0003, 0.1 |
| Model parameters | 1,180,416 |

| Batch | Best epoch | Best val loss | Test loss | Test PPL | Test Top-1 | Top-3 | Top-5 | Training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 4.077299 | 4.083347 | 59.343750 | 0.227214 | 0.374475 | 0.447636 | 714.121 s |
| 8 | 10 | 4.164609 | 4.173029 | 64.911758 | 0.217844 | 0.361816 | 0.434430 | 451.789 s |
| 16 | 10 | 4.223969 | 4.233643 | 68.968029 | 0.213138 | 0.355547 | 0.426621 | 312.575 s |

이 실행에서는 batch 4가 가장 낮은 validation loss를 보였고 batch 16이 가장 빨리 끝났습니다. 단일 seed, 단일 장비, 10 epochs의 교육용 비교이므로 일반적인 최적 batch size로 해석하지 않습니다.

- [Batch experiment dashboard](../artifacts/pretraining/batch_size_experiment_dashboard.png)
- [Validation loss comparison](../artifacts/pretraining/loss_comparison_val.png)
- [Raw summary table](../artifacts/pretraining/summary_by_batch_size.md)
- [Batch 4 metrics](../artifacts/pretraining/batch_4_metrics.json)
- [Batch 8 metrics](../artifacts/pretraining/batch_8_metrics.json)
- [Batch 16 metrics](../artifacts/pretraining/batch_16_metrics.json)

## 최종 설정

| 항목 | 값 |
| --- | ---: |
| vocabulary | 3,000 |
| context length | 128 |
| embedding dimension | 128 |
| Transformer layers | 6 |
| attention heads | 4 |
| learning rate | 5e-4 |
| batch size | 4 |
| dropout | 0.1 |
| epochs | 10 |

최종 실행은 train loss 3.653, validation loss 3.769를 기록했습니다. 초기와 최종 실행은 layer, learning rate와 batch size가 함께 달라 특정 설정 하나의 효과로 해석할 수 없습니다.

![최종 설정의 train과 validation loss](images/final-training-loss.png)

## NumPy 기초 구현의 원본 기록

MNIST 원본의 [`REPORT.md`](https://github.com/devhyun05/group4-mnist-lab/blob/main/REPORT.md)는 784→512→256→10 구조와 537,354 parameters, baseline test accuracy 98.44%를 기록합니다.

이 값은 원본 보고서의 historical result입니다. 현재 독립 저장소 검증은 전체 MNIST 재학습 대신 layer, optimizer와 network 단위 테스트를 실행합니다.

## 현재 CPU 재현

현재 환경에서는 `scripts/smoke_train.py`로 작은 GPT의 forward, next-token loss, backward와 AdamW update가 CPU에서 연결되는지 확인합니다.

| Field | Value |
| --- | --- |
| Run date | 2026-07-18 |
| Device | CPU |
| Torch | 2.13.0+cpu |
| Seed, steps | 42, 5 |
| Vocabulary, context | 64, 8 |
| Embedding, layers, heads | 16, 1, 4 |
| Initial loss | 4.3087930679 |
| Final loss | 2.6705155373 |

```text
4.308793 -> 3.898402 -> 3.537956 -> 3.219144 -> 2.937734 -> 2.670516
```

같은 synthetic batch를 반복 최적화한 결과입니다. 설치와 학습 경로의 연결은 확인하지만 언어 생성 품질이나 일반화 성능을 측정하지 않습니다.

2026-08-22 baseline에서 `uv run pytest -q`는 54개 테스트가 통과했습니다. Matplotlib의 non-interactive canvas 경고 2개가 있었지만 테스트 실패는 없었습니다.

## 생성과 fine-tuning 범위

과거 sample generation 표에는 encoding 문제로 깨진 문자열이 있습니다. [원본 표](../artifacts/pretraining/sample_generation_results.md)는 실행 흔적으로만 보존하고 생성 품질의 근거로 사용하지 않습니다.

`gpt/src/finetune.py`에는 sentiment dataset, classifier, train과 evaluate 도구가 있고 단위 테스트도 있습니다. 검증 가능한 accuracy 산출물은 없어 정확도 수치를 제시하지 않습니다.

## 재현 명령

```bash
uv sync
uv run pytest -q
uv run python scripts/smoke_train.py --output artifacts/current/smoke-result.json
```

## 해석 한계

- 발표 기록 중 layer, learning rate, dropout과 최종 실행에는 원본 지표 파일이 없습니다.
- batch 비교는 단일 seed와 단일 장비에서 실행했습니다.
- 과거 GPU 결과와 현재 CPU smoke는 데이터, 장치, 모델 크기와 목적이 다릅니다.
- CPU smoke의 loss 감소는 하나의 synthetic batch에 대한 결과이며 벤치마크가 아닙니다.
- sentiment fine-tuning은 구현과 단위 테스트 범위까지만 확인됩니다.
