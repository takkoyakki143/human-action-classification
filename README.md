# 프로젝트명

어떤 문제를 어떤 방법으로 해결했고, 어떤 결과를 얻었는지 1~2문장으로 소개합니다.

비교 기준과 선택한 방법, 핵심 성능 지표 또는 학습 시간 등 실제 확인한 정량적 결과를 포함합니다. 성능이 개선되지 않았다면 실험 결과를 바탕으로 내린 최종 판단을 작성합니다.

> 나중에 이력서·포트폴리오·면접에서 프로젝트를 설명할 수 있도록 작성합니다. 직접 수행한 작업과 주요 판단을 중심으로 정리하고, 상세 코드와 전체 실험 기록은 관련 파일로 연결합니다.

## 1. 프로젝트 개요

- **해결하려던 문제:** 어떤 상황에서 무엇을 해결하려고 했는지 작성합니다.
- **목표와 평가 기준:** 어떤 결과를 얻으려 했고, 무엇을 기준으로 확인했는지 작성합니다.

| 항목 | 내용 |
|---|---|
| 기간 | YYYY.MM.DD ~ YYYY.MM.DD |
| 사용 데이터 | 데이터셋 이름 및 출처(링크) |
| 주요 기술 | 실제 사용한 언어·라이브러리·도구 |

## 2. 프로젝트 구조

기본 환경 설정과 아래 트리 형식을 사용하되, 실제 존재하는 파일과 폴더에 맞게 작성합니다.

```text
project/
├── readme.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── models/
└── figures/
```

## 3. 설치 및 실행 방법

### 환경 설치

- Python 버전은 `.python-version`, 패키지와 버전은 `pyproject.toml` 및 `uv.lock`을 확인합니다.
- [uv 공식 설치 안내](https://docs.astral.sh/uv/getting-started/installation/)에 따라 uv를 설치합니다.
- 저장소를 내려받은 후 프로젝트 폴더에서 의존성을 설치합니다.

```bash
git clone <repository-url>
cd <repository-name>
uv sync
```

Notebook 실행에 필요한 `ipykernel`은 프로젝트 의존성에 포함합니다.

### 데이터 준비 및 실행

1. 데이터 준비 방법과 `data/raw/` 아래의 실제 파일·폴더 위치를 작성합니다.
2. VS Code에서 프로젝트 폴더와 Notebook을 엽니다.
3. 우측 상단의 **Select Kernel**에서 프로젝트 **`.venv`의 Python 환경**을 선택합니다.
4. 아래 순서대로 Notebook을 실행합니다.

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | 데이터 이해 및 분석 |
| 2 | `notebooks/02_preprocessing.ipynb` | 데이터 분할 및 전처리 |
| 3 | `notebooks/03_modeling.ipynb` | 모델링·실험·결과 검증 |

실제 파일명과 실행 순서에 맞게 수정합니다. 필요한 GPU 환경, 모델·데이터 다운로드, API 설정 등이 있다면 재현에 필요한 내용과 관련 파일 위치를 추가합니다.

별도 서비스나 대시보드를 실제로 구현했다면 해당 파일·실행 방법·결과 화면을 추가합니다.

## 4. 수행 내역

직접 수행한 작업과 주요 판단을 다음과 같이 정리했습니다.

| 작업 구분 | 직접 수행한 작업 | 근거 자료 |
|---|---|---|
| 데이터 구조·품질 점검 | 전체 2,031장의 파일 존재, 클래스 매핑, 분할 출처를 확인하고 모든 이미지를 열어 손상·색 모드·흑백 여부를 점검했습니다. 원본 파일명의 행동 이름과 라벨을 대조하여 라벨 연결 오류, 손상, RGB 외 이미지가 모두 0건임을 확인했습니다. `group_id`는 이미지마다 고유하여 사람 단위 분할 정보로 사용할 수 없음을 확인했습니다. | [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) 1·4번 섹션, [`figures/image_quality_table.csv`](figures/image_quality_table.csv), [`figures/eda_summary.csv`](figures/eda_summary.csv) |
| 클래스 분포 분석·평가 지표 선정 | Train·Validation은 클래스별 80/20장으로 균형이지만, Test는 최대 2.31배 불균형(승마 196장, 보트 노 젓기 85장)임을 확인했습니다. 가장 적은 클래스를 모두 틀려도 Accuracy는 6.9%p만 낮아지지만 Macro 지표는 약 12.5%p 낮아지는 점을 근거로 Macro F1을 핵심 지표로 정했습니다. | `01_eda.ipynb` 2번 섹션, [`figures/01_class_distribution.png`](figures/01_class_distribution.png) |
| 이미지 특성 분석·전처리 기준 수립 | 종횡비가 0.46~2.79로 다양하고 클래스별로 세로형(등반·점프·달리기)과 가로형(승마·활쏘기·보트)이 나뉨을 확인했습니다. Center Crop은 클래스 평균 원본 면적의 약 43~47%를 잘라내 가장자리의 노·활이 사라지는 사례가 있어, 종횡비를 유지하는 Resize + Padding(224×224, ImageNet 평균색 여백)을 기본 전처리로 정했습니다. | `01_eda.ipynb` 3·7번 섹션, [`figures/02_size_aspect.png`](figures/02_size_aspect.png), [`figures/06_crop_vs_pad.png`](figures/06_crop_vs_pad.png) |
| 중복·데이터 누수 점검 | 지각 해시(dHash)와 사전학습 ResNet18 특징 벡터의 코사인 유사도, 두 가지 방법으로 분할 간 중복을 점검했습니다. dHash 후보 4쌍은 육안 확인 결과 모두 다른 사진이었고, Test 사진별 최근접 학습 사진의 상위 쌍에서도 동일 사진은 없었습니다. 다만 같은 행사·촬영 세션의 사진이 Train과 Test에 나뉘어 있음을 발견하여, 고정된 분할은 유지하고 결과 해석의 한계로 기록했습니다. 육안 판정은 Notebook 코드에 기록하여 검토표를 재현 가능하게 생성했습니다. | `01_eda.ipynb` 5·9·10번 섹션, [`figures/04_near_duplicates.png`](figures/04_near_duplicates.png), [`figures/07_nn_similarity_hist.png`](figures/07_nn_similarity_hist.png), [`results/duplicate_review.csv`](results/duplicate_review.csv) |
| 오류 분석 가설 사전 정리 | 클래스별 샘플과 EDA 결과를 바탕으로 점프↔달리기 혼동(자세·배경 유사), Padding 여백 모양 의존, 저해상도에서 작은 원반 누락, 행사·세션 단서 의존의 네 가지 가설을 정리했습니다. Test 사진별 최근접 학습 사진 유사도를 저장해 두어, 최종 Test 평가 후 세션 단서 의존 여부를 확인할 수 있도록 했습니다. | `01_eda.ipynb` 6·9번 섹션, [`figures/05_class_samples.png`](figures/05_class_samples.png), [`results/test_nearest_trainval.csv`](results/test_nearest_trainval.csv) |

## 5. 문제 해결 과정

### 주요 접근

- 사용한 데이터나 자료의 핵심 특징과 처리 방법
- 선택한 분석·모델·구현 방법과 그 이유
- 결과를 검증한 방법

### 중요한 문제와 판단

기억해 둘 만한 사례 1~2개를 중심으로 작성합니다.

- **문제·관찰:** 무엇이 잘되지 않았거나 추가 확인이 필요했는가?
- **판단 근거:** 어떤 데이터나 결과를 보고 원인 또는 개선 가능성을 생각했는가?
- **수행한 작업:** 어떤 방법을 선택하고 무엇을 변경했는가?
- **판단의 변화:** 결과를 확인한 뒤 유지·수정·제외한 것은 무엇인가?

수치와 실험 비교는 다음 항목에 모으고, 여기서는 선택 이유와 해결 과정을 설명합니다.

## 6. 주요 실험 및 결과

대표적인 시도와 결과를 기록합니다. 모든 실행을 나열할 필요는 없으며, 비교 기준과 주요 변경·판단이 드러나면 됩니다.

### 학습 전략 비교

동일한 모델 구조(ResNet18)를 사용한 Training from Scratch와 Transfer Learning의 분류 성능 및 학습 효율을 비교합니다. Transfer Learning은 실제 실험한 전략과 설정을 기준으로 작성합니다. 여러 전략을 실험했다면 비교표에 열을 추가합니다.

| 비교 항목 | Training from Scratch | Transfer Learning (실제 선택한 전략) |
|---|---|---|
| 모델 구조 | ResNet18 | ResNet18 |
| 초기 가중치 | 무작위 초기화 | 사용한 사전학습 가중치 |
| 학습 범위 | 전체 레이어 | 실제 학습한 레이어 |
| 전체 파라미터 수 |  |  |
| 학습 가능 파라미터 수 |  |  |
| 학습 Epoch 수 |  |  |
| 학습 시간 |  |  |
| Validation Macro F1 |  |  |

실제 측정값을 기록하고, 필요하면 Accuracy, 클래스별 Precision·Recall·F1 등의 지표를 추가합니다. 학습 곡선을 참고하여 수렴 속도와 과적합 여부도 비교합니다.

데이터 분할, GPU 등 비교 조건과 주요 학습 설정을 함께 작성합니다. 학습 시간은 개별 학습 실행에 소요된 시간인지, 여러 번의 튜닝을 포함한 전체 소요 시간인지 구분합니다.

### 주요 실험 및 개선 결과

각 학습 전략의 기본 학습, 설정 변경(튜닝), 오류·VLM 분석을 바탕으로 한 개선 실험의 대표 결과를 정리합니다. 실제 수행한 실험에 맞게 행을 추가하거나 수정합니다.

| 시도 또는 방법 | 목적·변경 내용 | 검증 결과 | 해석 및 결정 |
|---|---|---|---|
| Training from Scratch — 기본 학습 |  |  |  |
| Training from Scratch — 튜닝 |  |  |  |
| Transfer Learning — 기본 학습 |  |  |  |
| Transfer Learning — 튜닝 |  |  |  |
| 오류 분석 기반 개선 실험 |  |  |  |

수치에는 평가 지표와 측정 조건을 함께 적습니다. 개선되지 않은 시도도 판단에 영향을 주었다면 남깁니다.

### 최종 결과 및 해석

- **최종 결과:** 선택한 방법, 핵심 성과와 확인한 범위를 정리합니다. Validation 데이터에서 모델을 선정하고, 최종 모델의 Test 평가 결과를 구분하여 작성합니다.
- **결과의 의미:** 초기 목표나 비교 기준에 비해 무엇이 달라졌는지 설명합니다.
- **근거 자료:** 관련 코드, 평가 결과, 시각화 또는 구현 결과의 위치를 연결합니다.

### 주요 결과 시각화

Notebook을 실행하지 않아도 프로젝트의 주요 결과와 판단을 확인할 수 있도록, 실제 분석 결과를 이미지로 저장하여 README에 첨부합니다.

아래는 이미지 첨부 방법의 예시입니다. 실제 생성한 이미지의 파일명과 경로를 사용하고, 각 이미지 아래에는 확인한 결과와 해석을 간단히 작성합니다.

```markdown
#### 학습 곡선

![Learning Curves](figures/learning_curves.png)

학습 전략에 따른 수렴 속도와 Train / Validation Loss의 차이를 설명합니다.

#### 최종 모델 평가

![Confusion Matrix](figures/confusion_matrix.png)

최종 모델에서 주로 발생한 분류 오류와 클래스별 성능을 설명합니다.

#### 대표 오분류 사례

![Error Examples](figures/error_examples.png)

대표 오분류 이미지의 정답·예측 클래스, 관찰한 특징과 VLM 분석을 통해 수립한 개선 가설을 설명합니다.
```

## 7. 한계 및 배운 점

- **한계:** 해결하지 못한 문제와 검증하지 못한 범위
- **배운 점:** 이번 경험을 통해 바뀐 판단 기준이나 작업 방식
- **다음 접근:** 비슷한 문제를 다시 맡는다면 먼저 확인하거나 다르게 시도할 것
