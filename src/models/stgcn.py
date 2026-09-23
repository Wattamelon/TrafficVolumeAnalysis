"""STGCN baseline 모델 구현.

이 모듈은 논문의 baseline인 Spatio-Temporal Graph Convolutional Network를
구현한다. 입력은 ``(배치, 9채널, 12시점, lane 수)``이며, 9채널은 네 차종의
queue length와 average speed(8개), 그리고 cyclic time-of-day encoding(1개)이다.
출력은 다음 5분 시점의 여덟 교통 특성으로 ``(배치, 8채널, lane 수)`` 형태다.

각 ST-Conv block은 시간 합성곱, lane adjacency matrix를 사용하는 graph
convolution, 두 번째 시간 합성곱 순서로 동작한다. 두 block 뒤의 출력 head가
12시점 축을 하나의 다음 시점 예측으로 압축한다.
"""

# PyTorch tensor 자료형과 신경망 모듈을 사용하기 위해 torch를 불러온다.
import torch
# 레이어를 만들기 위해 PyTorch의 neural network 모듈을 불러온다.
from torch import nn


class GraphConv(nn.Module):
    """인접한 lane의 특징을 한 번 전파하는 one-hop graph convolution.

    adjacency의 행 ``n``은 lane ``n``으로 들어오는 연결 가중치를 뜻한다.
    self-loop가 포함된 adjacency를 전달해야 자기 lane의 정보도 유지된다.
    """

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor):
        # 부모 nn.Module을 초기화해 하위 레이어와 parameter 등록을 가능하게 한다.
        super().__init__()
        # adjacency를 model state에 저장하고 model.to(device) 이동 시 함께 이동하게 한다.
        self.register_buffer("adjacency", adjacency.float())
        # lane별 전파 결과의 채널 수를 바꾸는 1x1 convolution을 생성한다.
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x의 shape은 (B, C, T, N)이고 adjacency의 shape은 (N, N)이다.
        # einsum으로 각 시간과 채널에서 adjacency @ lane_feature 연산을 수행한다.
        propagated = torch.einsum("nm,bctm->bctn", self.adjacency, x)
        # 전파된 특징에 1x1 convolution을 적용해 요청한 output channel 수로 변환한다.
        return self.projection(propagated)


class STConvBlock(nn.Module):
    """시간 합성곱 -> graph convolution -> 시간 합성곱으로 구성된 ST-Conv block."""

    def __init__(self, in_channels: int, out_channels: int, adjacency: torch.Tensor, kernel_size: int = 3):
        # 부모 nn.Module을 초기화한다.
        super().__init__()
        # same padding을 계산해 시간 축 길이(T=12)를 보존한다.
        padding = (kernel_size - 1) // 2
        # 첫 temporal convolution이 시간 인접 패턴을 추출한다.
        self.temporal_in = nn.Conv2d(in_channels, out_channels, (kernel_size, 1), padding=(padding, 0), bias=False)
        # 첫 temporal convolution의 출력에 graph convolution을 적용한다.
        self.graph = GraphConv(out_channels, out_channels, adjacency)
        # graph convolution 이후의 시간 패턴을 다시 정제하는 convolution을 생성한다.
        self.temporal_out = nn.Conv2d(out_channels, out_channels, (kernel_size, 1), padding=(padding, 0), bias=False)
        # 첫 temporal convolution 출력의 batch normalization을 생성한다.
        self.norm1 = nn.BatchNorm2d(out_channels)
        # graph convolution 출력의 batch normalization을 생성한다.
        self.norm2 = nn.BatchNorm2d(out_channels)
        # 두 번째 temporal convolution 출력의 batch normalization을 생성한다.
        self.norm3 = nn.BatchNorm2d(out_channels)
        # 각 stage 뒤에 적용할 비선형 활성화 함수를 생성한다.
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 첫 시간 convolution, normalization, ReLU를 순서대로 적용한다.
        x = self.activation(self.norm1(self.temporal_in(x)))
        # graph convolution, normalization, ReLU를 순서대로 적용한다.
        x = self.activation(self.norm2(self.graph(x)))
        # 두 번째 시간 convolution, normalization, ReLU를 적용한 결과를 반환한다.
        return self.activation(self.norm3(self.temporal_out(x)))


class STGCN(nn.Module):
    """두 개의 ST-Conv block으로 다음 한 시점을 예측하는 STGCN baseline.

    Args:
        adjacency: self-loop가 포함된 lane-to-lane adjacency matrix, shape ``(N, N)``.
        input_channels: 논문 입력 채널 수로 기본값은 9이다.
        output_channels: 네 차종 x 두 feature에 해당하는 기본값 8이다.
        hidden_channels: 각 ST-Conv block의 hidden channel 수로 논문 기본값은 64이다.
        history_steps: 입력 과거 시점 수로 논문 기본값은 12이다.
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        input_channels: int = 9,
        output_channels: int = 8,
        hidden_channels: int = 64,
        history_steps: int = 12,
    ):
        # 부모 nn.Module을 초기화한다.
        super().__init__()
        # 입력 검증과 출력 head 설정에 쓸 lane node 수를 저장한다.
        self.num_nodes = adjacency.shape[0]
        # 입력 시계열 길이를 저장해 forward에서 shape을 검증한다.
        self.history_steps = history_steps
        # 9개 입력 채널을 hidden channel로 바꾸는 첫 ST-Conv block을 생성한다.
        self.block1 = STConvBlock(input_channels, hidden_channels, adjacency)
        # hidden channel을 유지하며 더 높은 수준의 시공간 특징을 추출하는 두 번째 block을 생성한다.
        self.block2 = STConvBlock(hidden_channels, hidden_channels, adjacency)
        # 전체 12시점 축을 하나의 다음 시점으로 압축해 8개 target channel을 만든다.
        self.head = nn.Conv2d(hidden_channels, output_channels, (history_steps, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 입력이 4차원 tensor이고 시간 및 node 축이 모델 설정과 같은지 검사한다.
        if x.ndim != 4 or x.shape[2:] != (self.history_steps, self.num_nodes):
            # 잘못된 데이터 shape을 빠르게 찾을 수 있도록 기대 shape과 실제 shape을 알린다.
            raise ValueError(f"Expected (B, C, {self.history_steps}, {self.num_nodes}); received {tuple(x.shape)}")
        # 두 ST-Conv block을 통과시킨 후 head를 적용하고 크기 1의 시간 축을 제거한다.
        return self.head(self.block2(self.block1(x))).squeeze(2)
