"""저장된 checkpoint를 비공개 test data에서 평가하는 command-line 모듈.

이 모듈은 ``train.py``가 만든 ``best.pt``를 읽고 test NPZ 전체에서 prediction을
계산한다. baseline과 제안 모델 모두 같은 입력 형식과 MAE/RMSE 계산 함수를
사용하므로, 논문의 공정한 모델 비교 흐름을 코드 수준에서 보여 준다.
"""

# command-line argument를 읽기 위해 argparse를 불러온다.
import argparse

# checkpoint loading과 device 이동에 사용할 PyTorch를 불러온다.
import torch

# private data와 graph metadata를 읽는 함수를 불러온다.
from src.data import load_graph, make_loader
# MAE와 RMSE를 계산하는 공통 함수를 불러온다.
from src.metrics import regression_metrics
# checkpoint 구조에 맞는 baseline과 제안 모델 클래스를 불러온다.
from src.models import ClusterResidualSTGCN, STGCN


def main() -> None:
    """평가 인자를 읽고 checkpoint prediction의 MAE/RMSE를 출력한다."""

    # command-line parser 객체를 생성한다.
    parser = argparse.ArgumentParser()
    # checkpoint가 어떤 모델 구조인지 알려 주는 필수 인자를 추가한다.
    parser.add_argument("--model", choices=("stgcn", "residual-stgcn"), required=True)
    # private test NPZ 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--test", required=True)
    # lane adjacency matrix 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--adjacency", required=True)
    # lane별 cluster ID 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--cluster-ids", required=True)
    # train.py가 저장한 model state checkpoint 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--checkpoint", required=True)
    # 평가 batch size를 정의한다.
    parser.add_argument("--batch-size", type=int, default=8)
    # 사용자 입력을 Namespace 형태로 파싱한다.
    args = parser.parse_args()

    # CUDA가 있으면 GPU를, 없으면 CPU를 평가 device로 선택한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # private graph files에서 adjacency matrix와 cluster ID를 읽는다.
    adjacency, cluster_ids = load_graph(args.adjacency, args.cluster_ids)
    # 모델 이름에 맞는 uninitialized model instance를 생성한다.
    model = STGCN(adjacency) if args.model == "stgcn" else ClusterResidualSTGCN(adjacency, cluster_ids)
    # 안전한 weights-only 방식으로 checkpoint dictionary를 현재 device에 불러온다.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # checkpoint의 parameter 값을 생성한 모델 구조에 넣는다.
    model.load_state_dict(checkpoint["state_dict"])
    # model 전체를 evaluation device로 옮기고 evaluation mode로 전환한다.
    model.to(device).eval()

    # 모든 test batch의 prediction tensor를 저장할 빈 list를 만든다.
    predictions = []
    # 모든 test batch의 target tensor를 저장할 빈 list를 만든다.
    targets = []
    # gradient 계산을 끄고 inference memory를 줄인다.
    with torch.no_grad():
        # test dataset의 모든 batch를 순서대로 순회한다.
        for x, y, residuals in make_loader(args.test, args.batch_size, shuffle=False):
            # 입력과 residual sequence를 실행 device로 옮긴다.
            x, residuals = x.to(device), residuals.to(device)
            # baseline 또는 제안 모델에 맞는 forward를 수행한다.
            prediction = model(x) if args.model == "stgcn" else model(x, residuals)
            # GPU memory에 쌓이지 않게 prediction을 CPU tensor로 옮겨 저장한다.
            predictions.append(prediction.cpu())
            # target도 CPU tensor로 저장해 마지막에 전체 metrics를 계산한다.
            targets.append(y)
    # batch dimension으로 모든 prediction을 이어 붙이고 target도 같은 방식으로 합친다.
    print(regression_metrics(torch.cat(predictions), torch.cat(targets)))


# 이 파일이 import가 아니라 직접 실행됐을 때만 evaluation workflow를 시작한다.
if __name__ == "__main__":
    # command-line 평가 main 함수를 호출한다.
    main()
