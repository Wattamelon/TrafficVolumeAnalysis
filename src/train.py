"""STGCN baseline 또는 Cluster-LSTM residual model을 학습하는 command-line 모듈.

사용자는 private training/validation NPZ, adjacency NPY, cluster ID NPY의 경로를
인자로 전달한다. ``--model stgcn``은 residual sequence를 학습 입력으로 쓰지
않고 baseline만 학습한다. ``--model residual-stgcn``은 과거 12-step residual
sequence를 추가로 사용한다. 매 epoch 뒤 validation MSE가 가장 낮은 가중치를
``output/best.pt``에 저장한다.
"""

# command-line argument를 읽기 위해 argparse를 불러온다.
import argparse
# Python과 NumPy의 난수 시드를 고정하기 위해 random을 불러온다.
import random
# checkpoint 저장 경로를 다루기 위해 Path를 불러온다.
from pathlib import Path

# 난수 생성과 평균 계산에 사용할 NumPy를 불러온다.
import numpy as np
# tensor, optimizer, CUDA 설정에 사용할 PyTorch를 불러온다.
import torch
# loss function 및 모듈 type annotation에 사용할 nn을 불러온다.
from torch import nn

# 비공개 데이터를 읽고 graph tensor를 만드는 함수를 불러온다.
from src.data import load_graph, make_loader
# baseline과 제안 모델 클래스를 불러온다.
from src.models import ClusterResidualSTGCN, STGCN


def parse_args() -> argparse.Namespace:
    """학습에 필요한 command-line 인자를 정의하고 반환한다."""

    # command-line parser 객체를 생성한다.
    parser = argparse.ArgumentParser()
    # 어떤 모델을 학습할지 선택하는 필수 인자를 추가한다.
    parser.add_argument("--model", choices=("stgcn", "residual-stgcn"), required=True)
    # private training NPZ 파일 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--train", required=True, help="Private training NPZ: x, y, residuals arrays")
    # private validation NPZ 파일 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--valid", required=True, help="Private validation NPZ: x, y, residuals arrays")
    # lane adjacency matrix 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--adjacency", required=True)
    # lane별 connectivity cluster ID 경로를 받는 필수 인자를 추가한다.
    parser.add_argument("--cluster-ids", required=True)
    # checkpoint가 저장될 결과 폴더를 정의한다.
    parser.add_argument("--output", default="outputs")
    # 전체 학습 epoch 수를 정의한다.
    parser.add_argument("--epochs", type=int, default=40)
    # mini-batch 크기를 정의한다.
    parser.add_argument("--batch-size", type=int, default=8)
    # Adam optimizer의 learning rate를 정의한다.
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    # 재현성을 위한 random seed를 정의한다.
    parser.add_argument("--seed", type=int, default=42)
    # command-line에서 입력된 값을 Namespace로 변환해 반환한다.
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch, cuDNN의 난수 동작을 가능한 한 결정적으로 고정한다."""

    # Python 표준 라이브러리 난수 생성기의 seed를 고정한다.
    random.seed(seed)
    # NumPy 난수 생성기의 seed를 고정한다.
    np.random.seed(seed)
    # CPU PyTorch 난수 생성기의 seed를 고정한다.
    torch.manual_seed(seed)
    # CUDA를 사용할 수 있는 환경인지 확인한다.
    if torch.cuda.is_available():
        # 모든 CUDA device의 PyTorch 난수 생성기 seed를 고정한다.
        torch.cuda.manual_seed_all(seed)
    # cuDNN이 가능한 결정론적 알고리즘을 선택하도록 설정한다.
    torch.backends.cudnn.deterministic = True
    # 입력 shape에 따라 매번 다른 최적 kernel을 고르는 benchmark를 끈다.
    torch.backends.cudnn.benchmark = False


def predict(model: nn.Module, name: str, x: torch.Tensor, residuals: torch.Tensor) -> torch.Tensor:
    """선택된 모델 타입에 맞는 forward 호출을 하나의 함수로 통일한다."""

    # baseline은 traffic input만 받으므로 residuals 없이 model(x)를 호출한다.
    return model(x) if name == "stgcn" else model(x, residuals)


def mean_mse(model: nn.Module, name: str, loader, device: torch.device) -> float:
    """validation loader 전체의 batch MSE 평균을 계산한다."""

    # dropout과 batch normalization을 evaluation mode로 전환한다.
    model.eval()
    # 각 batch의 MSE 값을 저장할 빈 list를 만든다.
    losses = []
    # gradient 계산을 끄고 memory와 시간을 절약한다.
    with torch.no_grad():
        # validation loader의 모든 batch를 순회한다.
        for x, y, residuals in loader:
            # 입력과 residual sequence를 실행 device로 옮겨 예측한다.
            prediction = predict(model, name, x.to(device), residuals.to(device))
            # 예측과 target의 MSE를 계산해 Python float으로 list에 저장한다.
            losses.append(nn.functional.mse_loss(prediction, y.to(device)).item())
    # batch별 MSE 평균을 float로 변환해 반환한다.
    return float(np.mean(losses))


def main() -> None:
    """인자 파싱부터 학습, validation, best checkpoint 저장까지 수행한다."""

    # command-line 인자를 읽는다.
    args = parse_args()
    # 비교 가능한 결과를 위해 모든 난수 seed를 고정한다.
    set_seed(args.seed)
    # CUDA가 있으면 GPU를, 없으면 CPU를 학습 device로 선택한다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # private graph files에서 adjacency와 lane cluster ID를 읽는다.
    adjacency, cluster_ids = load_graph(args.adjacency, args.cluster_ids)
    # training NPZ를 shuffle이 켜진 DataLoader로 만든다.
    train_loader = make_loader(args.train, args.batch_size, shuffle=True)
    # validation NPZ를 순서를 보존하는 DataLoader로 만든다.
    valid_loader = make_loader(args.valid, args.batch_size, shuffle=False)
    # 선택한 이름에 따라 baseline 또는 residual correction 모델을 생성한다.
    model = STGCN(adjacency) if args.model == "stgcn" else ClusterResidualSTGCN(adjacency, cluster_ids)
    # 모델의 parameter와 buffer를 현재 실행 device로 옮긴다.
    model = model.to(device)
    # 논문의 학습 설정과 같은 Adam optimizer 및 weight decay를 생성한다.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # checkpoint를 저장할 output 경로 객체를 만든다.
    output = Path(args.output)
    # output 폴더가 없으면 parent folder까지 함께 생성한다.
    output.mkdir(parents=True, exist_ok=True)
    # 현재까지 가장 낮은 validation MSE를 무한대로 초기화한다.
    best = float("inf")

    # 지정한 epoch 수만큼 학습과 validation을 반복한다.
    for epoch in range(1, args.epochs + 1):
        # dropout과 batch normalization을 training mode로 전환한다.
        model.train()
        # training loader의 모든 mini-batch를 순회한다.
        for x, y, residuals in train_loader:
            # 이전 batch의 gradient를 0으로 초기화한다.
            optimizer.zero_grad()
            # 선택 모델의 forward 규칙에 맞춰 현재 batch prediction을 계산한다.
            prediction = predict(model, args.model, x.to(device), residuals.to(device))
            # target과 prediction의 mean squared error를 학습 loss로 계산한다.
            loss = nn.functional.mse_loss(prediction, y.to(device))
            # loss에서 각 parameter에 대한 gradient를 계산한다.
            loss.backward()
            # gradient 폭주를 줄이기 위해 global norm을 5.0으로 clip한다.
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            # optimizer가 계산된 gradient를 사용해 parameter를 한 번 갱신한다.
            optimizer.step()
        # 한 epoch 학습이 끝난 뒤 validation set MSE를 계산한다.
        validation_mse = mean_mse(model, args.model, valid_loader, device)
        # 진행 상황을 간결하게 terminal에 출력한다.
        print(f"epoch={epoch:03d} validation_mse={validation_mse:.6f}")
        # 이번 validation MSE가 지금까지 최고 성능보다 좋은지 검사한다.
        if validation_mse < best:
            # 새로운 최고 성능 값을 저장한다.
            best = validation_mse
            # 모델 종류와 state_dict만 담아 재사용 가능한 best checkpoint를 저장한다.
            torch.save({"model": args.model, "state_dict": model.state_dict()}, output / "best.pt")


# 이 파일이 import가 아니라 직접 실행됐을 때만 학습 main 함수를 호출한다.
if __name__ == "__main__":
    # 전체 학습 workflow를 시작한다.
    main()
