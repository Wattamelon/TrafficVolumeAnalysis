"""실제 비공개 교통 데이터 없이 공개 모델의 실행을 검증하는 예제.

작은 synthetic lane graph와 무작위 tensor를 생성해 STGCN baseline 및 제안
ClusterResidualSTGCN의 forward pass를 실행한다. 이 예제의 metric 값은 연구
성능을 뜻하지 않으며, 공개된 코드의 tensor shape과 호출 방식이 정상인지
확인하는 smoke test 용도다.
"""

# 실행 위치와 관계없이 프로젝트 root를 Python import path에 추가하기 위해 sys를 불러온다.
import sys
# 예제 파일 위치에서 프로젝트 root를 계산하기 위해 Path를 불러온다.
from pathlib import Path

# synthetic tensor와 adjacency matrix를 만들기 위해 PyTorch를 불러온다.
import torch

# examples 폴더의 부모인 프로젝트 root를 module search path 앞에 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# 예측값의 MAE와 RMSE를 출력하기 위해 metric 함수를 불러온다.
from src.metrics import regression_metrics
# baseline과 proposed model을 불러온다.
from src.models import ClusterResidualSTGCN, STGCN


def main() -> None:
    """작은 가짜 데이터에서 두 모델의 입력/출력 shape과 metric 계산을 확인한다."""

    # 예제를 반복 실행해도 동일한 난수 tensor가 생성되도록 seed를 고정한다.
    torch.manual_seed(42)
    # 실제 논문보다 작은 smoke-test용 batch 수, lane 수, cluster 수를 설정한다.
    batch, nodes, clusters = 2, 12, 4
    # 모든 lane이 자기 자신과 연결된 identity adjacency를 생성한다.
    adjacency = torch.eye(nodes)
    # 인접한 lane끼리도 연결되도록 상위 대각선에 edge를 추가한다.
    adjacency[torch.arange(nodes - 1), torch.arange(1, nodes)] = 1
    # 각 lane이 4개 중 하나의 connectivity cluster에 속하도록 ID를 생성한다.
    cluster_ids = torch.arange(nodes) % clusters
    # 논문 입력과 같은 9채널, 12시점 형태의 synthetic traffic input을 생성한다.
    x = torch.randn(batch, 9, 12, nodes)
    # 다음 시점의 8개 traffic feature target을 생성한다.
    target = torch.randn(batch, 8, nodes)
    # Cluster-LSTM에 전달할 12-step, 8-channel residual sequence를 생성한다.
    residuals = torch.randn(batch, 12, 8, nodes)

    # synthetic adjacency로 STGCN baseline instance를 생성한다.
    baseline = STGCN(adjacency)
    # synthetic adjacency와 cluster mapping으로 제안 모델 instance를 생성한다.
    proposed = ClusterResidualSTGCN(adjacency, cluster_ids)
    # baseline에 input만 전달해 next-step prediction을 계산한다.
    baseline_prediction = baseline(x)
    # 제안 모델에 input과 residual sequence를 전달해 corrected prediction을 계산한다.
    proposed_prediction = proposed(x, residuals)
    # baseline output shape과 synthetic target에 대한 smoke-test metric을 출력한다.
    print("Baseline:", baseline_prediction.shape, regression_metrics(baseline_prediction, target))
    # proposed output shape과 synthetic target에 대한 smoke-test metric을 출력한다.
    print("Proposed:", proposed_prediction.shape, regression_metrics(proposed_prediction, target))


# 이 파일이 import가 아니라 직접 실행됐을 때만 demo main 함수를 호출한다.
if __name__ == "__main__":
    # synthetic data smoke test를 시작한다.
    main()
