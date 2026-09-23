"""비공개 전처리 교통 데이터와 graph metadata를 읽는 모듈.

원본 lane-level traffic data는 공개하지 않는다. 대신 사용자는 ``.npz`` 파일에
입력 ``x``, target ``y``, 과거 residual sequence ``residuals``를 저장해 전달한다.
``x``는 ``(M, 9, 12, N)``, ``y``는 ``(M, 8, N)``, ``residuals``는
``(M, 12, 8, N)`` shape이어야 한다. graph adjacency와 cluster ID는 별도의
``.npy`` 파일로 전달한다. 자세한 형식은 ``docs/data_format.md``을 참고한다.
"""

# 경로 타입을 유연하게 받기 위해 pathlib의 Path를 불러온다.
from pathlib import Path

# NPZ와 NPY 파일을 읽기 위해 NumPy를 불러온다.
import numpy as np
# NumPy 배열을 PyTorch tensor로 바꾸기 위해 torch를 불러온다.
import torch
# PyTorch Dataset과 DataLoader를 만들기 위한 클래스를 불러온다.
from torch.utils.data import DataLoader, Dataset


class TrafficWindowDataset(Dataset):
    """사전 생성된 window 배열을 한 sample씩 제공하는 PyTorch Dataset."""

    def __init__(self, path: str | Path):
        # NPZ 파일을 열어 저장된 배열 이름과 내용을 읽는다.
        arrays = np.load(path)
        # 제안 모델 학습에 반드시 필요한 세 배열의 이름을 정의한다.
        required = {"x", "y", "residuals"}
        # 실제 파일에 없는 필수 배열 이름을 계산한다.
        missing = required.difference(arrays.files)
        # 하나라도 없으면 학습 시점보다 이른 단계에서 원인을 알린다.
        if missing:
            # 누락된 배열 이름을 정렬해 오류 메시지로 반환한다.
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        # 입력 배열을 float tensor로 변환한다.
        self.x = torch.from_numpy(arrays["x"]).float()
        # 다음 5분 target 배열을 float tensor로 변환한다.
        self.y = torch.from_numpy(arrays["y"]).float()
        # 과거 residual sequence 배열을 float tensor로 변환한다.
        self.residuals = torch.from_numpy(arrays["residuals"]).float()
        # 세 배열이 각각 논문에서 쓰는 차원 수를 가지는지 검사한다.
        if self.x.ndim != 4 or self.y.ndim != 3 or self.residuals.ndim != 4:
            # 잘못된 array layout을 바로 확인할 수 있도록 기대 shape을 알린다.
            raise ValueError("Expected x=(M,C,T,N), y=(M,C_out,N), residuals=(M,T,C_out,N).")
        # 세 배열이 같은 sample 수 M을 가지는지 검사한다.
        if not (len(self.x) == len(self.y) == len(self.residuals)):
            # sample alignment가 깨진 데이터를 거부한다.
            raise ValueError("x, y, and residuals must contain the same number of samples.")

    def __len__(self) -> int:
        # DataLoader가 epoch당 sample 수를 알 수 있도록 입력 배열의 길이를 반환한다.
        return len(self.x)

    def __getitem__(self, index: int):
        # 지정한 sample index의 입력, target, residual sequence를 함께 반환한다.
        return self.x[index], self.y[index], self.residuals[index]


def load_graph(adjacency_path: str | Path, cluster_ids_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """비공개 adjacency matrix와 lane별 cluster ID를 읽고 기본 shape을 검증한다."""

    # NPY adjacency matrix를 float tensor로 변환한다.
    adjacency = torch.from_numpy(np.load(adjacency_path)).float()
    # NPY lane cluster ID 배열을 long tensor로 변환한다.
    cluster_ids = torch.from_numpy(np.load(cluster_ids_path)).long()
    # adjacency가 N x N 정사각 행렬인지 검사한다.
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        # graph 정의가 잘못된 경우를 명확히 알린다.
        raise ValueError("adjacency must be a square matrix.")
    # cluster ID가 adjacency의 node 수 N개와 정확히 대응하는지 검사한다.
    if cluster_ids.shape != (adjacency.shape[0],):
        # lane mapping의 길이 불일치를 알린다.
        raise ValueError("cluster_ids must have one value per adjacency node.")
    # 검증된 graph tensor와 cluster tensor를 함께 반환한다.
    return adjacency, cluster_ids


def make_loader(path: str | Path, batch_size: int, shuffle: bool) -> DataLoader:
    """NPZ 데이터셋을 읽어 학습 또는 평가에 쓸 DataLoader를 생성한다."""

    # Dataset을 생성하고 요청한 batch 크기와 shuffle 설정으로 감싼다.
    return DataLoader(TrafficWindowDataset(path), batch_size=batch_size, shuffle=shuffle)
