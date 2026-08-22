# 기여 기록

이 문서는 이시원(`NearthYou`)이 작성자로 표시된 원본 commit을 source file과 현재 위치에 연결합니다. 독립 저장소에는 원본 Git history가 없으므로 기여 범위는 commit 링크와 원본 저장소를 기준으로 확인합니다.

## 주장 경계

- BPE, embedding, causal/multi-head attention, GPT backbone을 포함한 팀 구현 전체를 개인 단독 구현으로 주장하지 않습니다.
- 아래 표는 Git author가 이시원 계정으로 확인되는 commit만 개인 기여로 분리합니다.
- 원본에서 선별하지 않은 실험 스크립트와 보고서는 `현재 위치`에 `미이관`으로 표시합니다.

## GPT lab 개인 commit

원본: [`Soldbone/gpt-lab`](https://github.com/Soldbone/gpt-lab)

| Commit | 원본 file | 현재 위치 | 확인되는 동작 | Contributor |
| --- | --- | --- | --- | --- |
| [`853bdab`](https://github.com/Soldbone/gpt-lab/commit/853bdab) | `src/train.py` | `gpt/src/train.py` | single batch next-token loss | 이시원 (`NearthYou`) |
| [`e646620`](https://github.com/Soldbone/gpt-lab/commit/e646620) | `src/train.py` | `gpt/src/train.py` | DataLoader 평균 loss | 이시원 (`NearthYou`) |
| [`5ebd6f0`](https://github.com/Soldbone/gpt-lab/commit/5ebd6f0) | `src/train.py` | `gpt/src/train.py` | model/optimizer checkpoint 저장 | 이시원 (`NearthYou`) |
| [`1d1eeb5`](https://github.com/Soldbone/gpt-lab/commit/1d1eeb5) | `src/train.py` | `gpt/src/train.py` | checkpoint 복원과 epoch/step recovery | 이시원 (`NearthYou`) |
| [`a7f6735`](https://github.com/Soldbone/gpt-lab/commit/a7f6735) | `src/train.py` | `gpt/src/train.py` | temperature/top-k token generation | 이시원 (`NearthYou`) |
| [`bf6fcff`](https://github.com/Soldbone/gpt-lab/commit/bf6fcff) | `src/train.py` | `gpt/src/train.py` | encode→generate→decode sample 출력 | 이시원 (`NearthYou`) |
| [`8eb0f58`](https://github.com/Soldbone/gpt-lab/commit/8eb0f58) | `src/train.py` | `gpt/src/train.py` | pretraining loop, evaluation, checkpoint 연결 | 이시원 (`NearthYou`) |
| [`451669b`](https://github.com/Soldbone/gpt-lab/commit/451669b) | `src/train.py`, `tests/test_train.py` | `gpt/src/train.py`, `gpt/tests/test_train.py` | generation/training 경계 수정과 regression test | 이시원 (`NearthYou`) |
| [`f8c3ef6`](https://github.com/Soldbone/gpt-lab/commit/f8c3ef6) | `src/finetune.py` | `gpt/src/finetune.py` | sentiment dataset, classifier, train/evaluate utility | 이시원 (`NearthYou`) |
| [`cb69bdf`](https://github.com/Soldbone/gpt-lab/commit/cb69bdf) | experiment/reproducibility scripts and tests | 미이관; `docs/results.md`에서 provenance만 참조 | experiment output과 reproducibility test 보강 | 이시원 (`NearthYou`) |
| [`4fe533e`](https://github.com/Soldbone/gpt-lab/commit/4fe533e) | `outputs/batch_size_experiment_20260603_170551/` | `artifacts/pretraining/` | batch-size run config, metrics, graph, table 보존 | 이시원 (`NearthYou`) |

## NumPy와 MNIST lab 개인 commit

원본: [`devhyun05/group4-mnist-lab`](https://github.com/devhyun05/group4-mnist-lab)

| Commit | 원본 file | 현재 위치 | 확인되는 동작 | Contributor |
| --- | --- | --- | --- | --- |
| [`c2f49f7`](https://github.com/devhyun05/group4-mnist-lab/commit/c2f49f7) | `src/activations.py` | `foundations/mnist_numpy/src/activations.py` | ReLU, Softmax forward/backward | 이시원 (`NearthYou`) |
| [`af3579a`](https://github.com/devhyun05/group4-mnist-lab/commit/af3579a) | `src/layers.py` | `foundations/mnist_numpy/src/layers.py` | Affine forward/backward | 이시원 (`NearthYou`) |
| [`f5b2afb`](https://github.com/devhyun05/group4-mnist-lab/commit/f5b2afb) | `src/losses.py` | `foundations/mnist_numpy/src/losses.py` | mean cross-entropy loss | 이시원 (`NearthYou`) |
| [`624fbb3`](https://github.com/devhyun05/group4-mnist-lab/commit/624fbb3) | `src/optimizers.py` | `foundations/mnist_numpy/src/optimizers.py` | SGD parameter update | 이시원 (`NearthYou`) |
| [`539c5c0`](https://github.com/devhyun05/group4-mnist-lab/commit/539c5c0) | `src/optimizers.py` | `foundations/mnist_numpy/src/optimizers.py` | Adam moment update와 bias correction | 이시원 (`NearthYou`) |
| [`74b26f7`](https://github.com/devhyun05/group4-mnist-lab/commit/74b26f7) | `src/activations.py`, `src/layers.py` | `foundations/mnist_numpy/src/activations.py`, `layers.py` | Dropout과 BatchNorm | 이시원 (`NearthYou`) |
| [`5654845`](https://github.com/devhyun05/group4-mnist-lab/commit/5654845) | `src/network.py` | `foundations/mnist_numpy/src/network.py` | 784→512→256→10 network composition | 이시원 (`NearthYou`) |
| [`a2bfb80`](https://github.com/devhyun05/group4-mnist-lab/commit/a2bfb80) | `src/training.py` | `foundations/mnist_numpy/src/training.py` | mini-batch training, evaluation, loss plot | 이시원 (`NearthYou`) |
| [`3057901`](https://github.com/devhyun05/group4-mnist-lab/commit/3057901) | ablation report/script/CSV/plots | 미이관; `docs/results.md`에서 원본 report만 링크 | Dropout/BatchNorm ablation 초안과 artifact | 이시원 (`NearthYou`) |
| [`6bcc32a`](https://github.com/devhyun05/group4-mnist-lab/commit/6bcc32a) | ablation report/script/CSV/plots | 미이관; 원본 commit 참조 | full-MNIST ablation 실행 결과 반영 | 이시원 (`NearthYou`) |

## 검증 명령

```bash
uv run pytest foundations/mnist_numpy/tests gpt/tests -q
uv run pytest tests/test_smoke_train.py tests/test_evidence_contract.py -q
```

이 기록은 누가 어떤 commit을 작성했는지 보여줍니다. 해당 file의 모든 줄을 한 사람이 만들었다는 뜻은 아니므로 이후 team merge와 수정은 원본 history에서 함께 확인해야 합니다.
