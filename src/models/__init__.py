"""논문에서 사용한 예측 모델을 모아 둔 하위 패키지.

``STGCN``은 공간적 lane 연결성과 시간 변화를 동시에 학습하는 baseline이다.
``ClusterResidualSTGCN``은 baseline 예측에 cluster-level LSTM 잔차 보정을
더한 제안 모델이다.
"""

# 제안 모델 클래스를 외부 사용자가 ``src.models``에서 바로 불러올 수 있게 import한다.
from .residual_stgcn_lstm import ClusterResidualSTGCN
# baseline 모델 클래스를 외부 사용자가 ``src.models``에서 바로 불러올 수 있게 import한다.
from .stgcn import STGCN

# 패키지 외부에 공개할 이름을 명시해 의도하지 않은 내부 객체의 노출을 막는다.
__all__ = ["ClusterResidualSTGCN", "STGCN"]
