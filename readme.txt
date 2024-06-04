<git bash 사용법>

1.1. 데이터 다운로드

scripts/download_data.sh 스크립트를 실행하여 Kaggle에서 데이터를 다운로드합니다.

cd scripts
bash download_data.sh

이 명령은 kaggle.json 파일을 사용하여 Kaggle API를 통해 데이터를 다운로드하고 압축을 해제하여 data 디렉토리에 저장합니다.

1.2. 데이터 전처리

데이터를 전처리하여 훈련, 검증, 테스트 데이터셋으로 나눕니다.

python preprocess_data.py

이 명령은 data/LSWMD.pkl 파일을 읽고 전처리한 후, data/dataset_train.pickle, data/dataset_val.pickle, data/dataset_test.pickle 파일로 저장합니다.

2. 모델 훈련

모델을 훈련하려면 scripts/train_models.py 스크립트를 실행합니다.

python train_models.py

이 명령은 훈련 데이터셋과 검증 데이터셋을 로드하고, 모델을 훈련시키고, 검증하며, 최고의 모델을 save_best_model 디렉토리에 저장합니다.

3. 앙상블 모델 평가

훈련된 모델을 사용하여 앙상블 모델을 평가하려면 scripts/evaluate_ensemble.py 스크립트를 실행합니다.

python evaluate_ensemble.py

이 명령은 최상의 모델을 로드하고, 검증 데이터셋을 사용하여 앙상블 모델을 평가하고 성능을 출력합니다.

<colab 사용법>

1. Kaggle API 설정

!pip install kaggle
from google.colab import files
files.upload()  # 이 부분에서 본인의 kaggle.json 파일을 업로드하시면 됩니다.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

2. 데이터 다운로드

!kaggle datasets download -d qingyi/wm811k-wafer-map
!unzip wm811k-wafer-map.zip -d data/

# 3. 필요한 패키지 설치
!pip install efficientnet_pytorch
!pip install seaborn

4. 프로젝트 파일 가져오기

!git clone https://github.com/your-username/your-repo-name.git
%cd your-repo-name

5. 데이터 전처리

!python scripts/preprocess_data.py

6. 모델 훈련

!python scripts/train_models.py

7. 앙상블 평가

!python scripts/evaluate_ensemble.py
