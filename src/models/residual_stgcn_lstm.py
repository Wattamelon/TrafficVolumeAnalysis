"""논문의 STGCN + Cluster-LSTM residual correction 제안 모델 구현.

STGCN은 lane network의 공간 연결성과 최근 12개 시점의 패턴을 사용해 base
prediction을 만든다. 제안 모델은 여기에 STGCN이 남긴 과거 residual sequence를
더해 보정한다. 1,370개 lane에 LSTM을 각각 적용하지 않고, 물리적 lane 연결성을
기준으로 만든 439개 cluster에 average pooling을 적용한다. 각 cluster의 residual
sequence를 LSTM이 처리하고, 예측한 correction은 해당 cluster의 모든 lane으로
broadcast된다.

``forward``의 residual_sequence는 사전 계산된 ``ground truth - base prediction``
값이며 shape은 ``(B, 12, 8, N)``이다. 실제 데이터와 cluster mapping은 공개하지
않으므로, 입력 형식은 ``docs/data_format.md``에서 별도로 설명한다.
"""

# Tensor 연산과 타입 표기를 위해 PyTorch를 불러온다.
import torch
# LSTM, Linear, Dropout 등 신경망 레이어를 사용하기 위해 nn을 불러온다.
from torch import nn

# 같은 패키지의 STGCN baseline을 base predictor로 사용하기 위해 상대 import한다.
from .stgcn import STGCN


class ClusterResidualLSTM(nn.Module):
    """각 cluster의 residual sequence에서 하나의 8-channel correction을 예측한다."""

    def __init__(self, channels: int, hidden_channels: int = 64, dropout: float = 0.2):
        # 부모 nn.Module을 초기화한다.
        super().__init__()
        # residual feature를 LSTM hidden dimension으로 투영하는 fully connected layer를 생성한다.
        self.input_projection = nn.Linear(channels, hidden_channels)
        # cluster마다 독립된 sequence로 처리할 LSTM을 생성한다.
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        # 마지막 hidden state에 dropout을 적용해 과적합을 줄인다.
        self.dropout = nn.Dropout(dropout)
        # LSTM hidden dimension을 다시 교통 feature channel 수로 되돌리는 layer를 생성한다.
        self.output_projection = nn.Linear(hidden_channels, channels)
        # 초기 correction이 불필요하게 치우치지 않도록 output bias를 0으로 초기화한다.
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, cluster_residuals: torch.Tensor) -> torch.Tensor:
        # 입력 shape (B, T, K, C)를 각 cluster 독립 sequence가 되도록 분해한다.
        batch, steps, clusters, channels = cluster_residuals.shape
        # (B, T, K, C)를 (B*K, T, C)로 바꿔 하나의 LSTM batch로 효율적으로 처리한다.
        sequence = cluster_residuals.permute(0, 2, 1, 3).reshape(batch * clusters, steps, channels)
        # 각 시간의 C개 residual feature를 hidden dimension으로 변환한다.
        sequence = self.input_projection(sequence)
        # LSTM으로 각 cluster의 시간적 residual 패턴을 학습한다.
        sequence, _ = self.lstm(sequence)
        # 마지막 time step hidden state에서 C개 보정값을 예측한다.
        correction = self.output_projection(self.dropout(sequence[:, -1]))
        # (B*K, C)를 원래 cluster 축을 포함한 (B, K, C)로 복원한다.
        return correction.reshape(batch, clusters, channels)


class ClusterResidualSTGCN(nn.Module):
    """STGCN base prediction에 cluster-level LSTM correction을 더하는 모델.

    ``cluster_ids``의 각 원소는 lane 하나가 속한 connectivity-based cluster 번호다.
    논문에서는 ``N=1,370``이고 ``K=439``이다. 이 구현은 더 작은 synthetic graph도
지원하므로 공개 저장소에서 데이터 없이 모델 구조를 검증할 수 있다.
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        cluster_ids: torch.Tensor,
        input_channels: int = 9,
        output_channels: int = 8,
        hidden_channels: int = 64,
        history_steps: int = 12,
    ):
        # 부모 nn.Module을 초기화한다.
        super().__init__()
        # lane 수와 동일한 길이의 1차원 cluster ID가 전달됐는지 검사한다.
        if cluster_ids.ndim != 1 or cluster_ids.numel() != adjacency.shape[0]:
            # cluster mapping이 graph와 맞지 않으면 후속 pooling 결과가 잘못되므로 즉시 중단한다.
            raise ValueError("cluster_ids must be a one-dimensional value for each lane node.")
        # cluster 번호는 0 이상의 정수여야 하므로 음수 ID를 거부한다.
        if cluster_ids.min() < 0:
            # 잘못된 cluster ID를 명확히 알린다.
            raise ValueError("cluster_ids must be non-negative integers.")
        # STGCN baseline을 제안 모델의 base prediction branch로 생성한다.
        self.base = STGCN(adjacency, input_channels, output_channels, hidden_channels, history_steps)
        # residual tensor의 channel 검증에 쓸 target feature 수를 저장한다.
        self.output_channels = output_channels
        # residual sequence의 길이를 검증하기 위해 history step 수를 저장한다.
        self.history_steps = history_steps
        # 가장 큰 ID에 1을 더해 실제 cluster 개수 K를 계산한다.
        self.num_clusters = int(cluster_ids.max().item()) + 1
        # cluster ID를 model state에 저장하고 CPU/GPU 이동 시 함께 이동하게 한다.
        self.register_buffer("cluster_ids", cluster_ids.long())
        # cluster residual sequence를 correction으로 바꾸는 LSTM branch를 생성한다.
        self.residual_lstm = ClusterResidualLSTM(output_channels, hidden_channels)

    def _aggregate_clusters(self, residuals: torch.Tensor) -> torch.Tensor:
        # 입력 residuals의 shape을 풀어 각 축을 명시적으로 사용한다.
        batch, steps, channels, nodes = residuals.shape
        # residual tensor의 lane 수가 cluster mapping의 lane 수와 같은지 검사한다.
        if nodes != self.cluster_ids.numel():
            # graph와 residual 데이터가 다른 경우를 즉시 알린다.
            raise ValueError("Residual sequence node count does not match cluster_ids.")
        # 평균 pooling 결과를 저장할 (B, T, K, C) tensor를 입력과 같은 device/type으로 생성한다.
        pooled = residuals.new_zeros(batch, steps, self.num_clusters, channels)
        # 각 connectivity cluster에 대해 lane별 residual 평균을 계산한다.
        for cluster in range(self.num_clusters):
            # 현재 cluster에 포함된 lane을 고르는 boolean mask를 만든다.
            mask = self.cluster_ids == cluster
            # 선택된 lane 축의 평균을 내어 해당 cluster의 residual sequence를 채운다.
            pooled[:, :, cluster] = residuals[:, :, :, mask].mean(dim=-1)
        # cluster-level residual sequence를 반환한다.
        return pooled

    def _broadcast_clusters(self, corrections: torch.Tensor) -> torch.Tensor:
        # corrections (B, K, C)에서 각 lane의 cluster ID에 맞는 correction을 선택한다.
        lane_corrections = corrections[:, self.cluster_ids]
        # 모델 출력과 같은 (B, C, N) 순서로 축을 바꾸고 memory를 연속적으로 만든다.
        return lane_corrections.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor, residual_sequence: torch.Tensor) -> torch.Tensor:
        # residual sequence의 시간 및 feature 축이 논문 설정과 같은지 검사한다.
        if residual_sequence.shape[1:3] != (self.history_steps, self.output_channels):
            # 올바른 shape과 실제 shape을 표시해 전처리 오류를 찾기 쉽게 한다.
            raise ValueError(
                f"Expected residual_sequence shape (B, {self.history_steps}, {self.output_channels}, N); "
                f"received {tuple(residual_sequence.shape)}"
            )
        # 현재 12-step traffic input으로 STGCN의 base next-step prediction을 계산한다.
        base_prediction = self.base(x)
        # node-level residual sequence를 439개 cluster-level sequence로 평균 pooling한다.
        cluster_residuals = self._aggregate_clusters(residual_sequence)
        # 각 cluster의 시간적 residual pattern에서 correction vector를 예측한다.
        cluster_correction = self.residual_lstm(cluster_residuals)
        # cluster correction을 lane별로 broadcast해 base prediction에 더한 최종 예측을 반환한다.
        return base_prediction + self._broadcast_clusters(cluster_correction)
