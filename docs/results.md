# Results and evidence

이 문서는 과거 원본 저장소에 보존된 실험과 현재 환경에서 새로 실행한 검증을 분리합니다. 두 결과는 설정과 목적이 달라 동일한 성능 실험으로 비교하지 않습니다.

## Historical result — commit `4fe533e`

Source of Truth는 [`Soldbone/gpt-lab` commit `4fe533e`](https://github.com/Soldbone/gpt-lab/commit/4fe533e)와 그 commit의 `outputs/batch_size_experiment_20260603_170551` 디렉터리입니다. 이 저장소의 `artifacts/pretraining/` 파일은 해당 경로에서 선택 추출했고, 추출 전후 SHA-256 일치를 확인했습니다.

### Run configuration

| Field | Value |
| --- | --- |
| Run date | 2026-06-03 |
| Seed | 42 |
| Device | CUDA (`cuda_available: true`) |
| Epochs | 10 |
| Corpus size | 1,500,000 characters |
| Vocabulary / context | 3,000 / 128 |
| Embedding / layers / heads | 128 / 2 / 4 |
| Dropout | 0.1 |
| Learning rate / weight decay | 0.0003 / 0.1 |
| Model parameters | 1,180,416 |

전체 설정은 [run_config.json](../artifacts/pretraining/run_config.json)에 있습니다.

### Batch-size comparison

| Batch | Best epoch | Best val loss | Test loss | Test PPL | Test Top-1 | Top-3 | Top-5 | Training time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 10 | 4.077299 | 4.083347 | 59.343750 | 0.227214 | 0.374475 | 0.447636 | 714.121 s |
| 8 | 10 | 4.164609 | 4.173029 | 64.911758 | 0.217844 | 0.361816 | 0.434430 | 451.789 s |
| 16 | 10 | 4.223969 | 4.233643 | 68.968029 | 0.213138 | 0.355547 | 0.426621 | 312.575 s |

- [Batch-size experiment dashboard](../artifacts/pretraining/batch_size_experiment_dashboard.png)
- [Validation-loss comparison](../artifacts/pretraining/loss_comparison_val.png)
- [Raw summary table](../artifacts/pretraining/summary_by_batch_size.md)
- [Per-epoch metrics: batch 4](../artifacts/pretraining/batch_4_metrics.json), [batch 8](../artifacts/pretraining/batch_8_metrics.json), [batch 16](../artifacts/pretraining/batch_16_metrics.json)

이 결과에서는 작은 batch가 더 낮은 validation loss를 보였고 큰 batch가 더 짧은 total training time을 보였습니다. 단일 seed·단일 장비·10 epochs의 교육용 비교이므로 일반적인 최적 batch-size 결론으로 확장하지 않습니다.

### Generation and fine-tuning status

과거 run은 sample generation 표도 남겼지만 일부 입력·출력 문자열이 당시 인코딩 문제로 깨져 있습니다. 따라서 [원본 표](../artifacts/pretraining/sample_generation_results.md)는 실행 흔적으로만 보존하며 생성 품질의 근거로 사용하지 않습니다. 이는 vocabulary 3,000, embedding 128, 2-layer인 작은 교육용 모델이기도 합니다.

감성 분류용 dataset/classifier/train/evaluate 구현과 unit test는 `gpt/src/finetune.py`에 있습니다. 그러나 commit history에 외부에서 검증 가능한 fine-tuning accuracy artifact가 보존되지 않아 정확도 수치를 제시하지 않습니다.

## NumPy foundation source-recorded result

MNIST 원본의 [`REPORT.md`](https://github.com/devhyun05/group4-mnist-lab/blob/main/REPORT.md)는 784→512→256→10 구조, 537,354 parameters의 baseline test accuracy 98.44%를 기록합니다. 이는 원본 보고서에 기록된 historical result이며, 현재 standalone 검증은 full MNIST 재학습 대신 layer/network unit test를 실행합니다.

## Current reproduction — 2026-07-18

현재 환경에서는 `scripts/smoke_train.py`를 실행해 작은 GPT의 forward, next-token loss, backward, AdamW update가 CPU에서 연결되는지 확인했습니다.

| Field | Value |
| --- | --- |
| Device | CPU |
| Python | 3.11.15 |
| Torch | 2.13.0+cpu |
| Seed / steps | 42 / 5 |
| Vocabulary / context | 64 / 8 |
| Embedding / layers / heads | 16 / 1 / 4 |
| Initial loss | 4.3087930679 |
| Final loss | 2.6705155373 |

Loss curve:

```text
4.308793 -> 3.898402 -> 3.537956 -> 3.219144 -> 2.937734 -> 2.670516
```

원본 결과는 [artifacts/current/smoke-result.json](../artifacts/current/smoke-result.json)에 있습니다. 같은 synthetic batch를 반복 최적화해 loss 감소를 확인하는 연결성 테스트이므로 language quality나 generalization을 측정하지 않습니다.

## Reproduction commands

```bash
uv sync
uv run pytest -q
uv run python scripts/smoke_train.py --output artifacts/current/smoke-result.json
```

## Limits on interpretation

- Historical result와 Current reproduction은 data, device, model size, steps가 다릅니다.
- batch-size 표는 단일 seed 결과이며 통계적 유의성을 평가하지 않았습니다.
- current smoke loss 감소는 한 synthetic batch에 대한 것이며 benchmark가 아닙니다.
- sentiment fine-tuning은 구현·unit test 범위까지만 증명합니다.
