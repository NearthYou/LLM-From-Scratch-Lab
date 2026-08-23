# Attribution and reuse status

이 저장소는 다음 두 팀 학습 저장소의 코드를 검증 가능한 학습 흐름으로 재구성했습니다.

- GPT: [`Soldbone/gpt-lab`](https://github.com/Soldbone/gpt-lab)
- NumPy/MNIST: [`devhyun05/group4-mnist-lab`](https://github.com/devhyun05/group4-mnist-lab)
- GPT 개인 이력 보존 fork: [`NearthYou/gpt-lab`](https://github.com/NearthYou/gpt-lab)

## 기여자와 소유권 표시

원본 과제의 팀 기여자는 이현성, 이지섭, 양은열, 이시원입니다. BPE, embedding, causal/multi-head attention, GPT backbone을 포함한 팀 코드 전체를 이시원 개인의 단독 구현으로 표시하지 않습니다. Git history에서 이시원(`NearthYou`)의 commit으로 확인되는 범위는 [docs/contribution-map.md](docs/contribution-map.md)에 별도로 기록했습니다.

## 재구성과 공개 동의

이시원은 원본 팀 기여자들로부터 학습 결과를 별도 저장소로 재구성하고 공개하는 데 대한 동의를 확인했다고 밝혔습니다. 이 문서는 그 확인 사실과 출처를 기록하기 위한 것이며, 각 기여자의 저작권을 이전하거나 새로운 라이선스를 만드는 문서가 아닙니다.

원본 저장소와 이 저장소에는 팀이 승인한 별도 LICENSE가 없으므로, 공개 열람만으로 복제, 수정, 재배포 권한이 일반적으로 부여된다고 해석하지 않습니다. 제3자가 코드를 재사용하려면 원본 기여자에게 필요한 허락을 별도로 확인해야 합니다. 이 안내는 법률 자문이 아니라 보수적인 재사용 경계 표시입니다.

## 이 저장소에서 보존한 범위

- 최신 원본 구현을 기반으로 정리한 NumPy/MNIST 및 GPT source와 기능 테스트
- GPT 원본 commit `4fe533e`에서 SHA-256 일치를 확인해 추출한 선택적 실험 아티팩트
- 현재 환경에서 새로 실행한 deterministic CPU smoke 결과
- 원본 Git commit을 기준으로 작성한 개인 기여 map

원본 Git history는 이 저장소로 복제하지 않았습니다. 기여 확인의 Source of Truth는 위 원본 저장소와 명시된 commit hash입니다.
