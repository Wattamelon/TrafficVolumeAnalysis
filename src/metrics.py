"""교통량 예측 결과를 비교할 때 사용하는 회귀 평가 지표 모듈.

논문 README의 핵심 비교 지표인 MAE와 RMSE를 계산한다. 입력 tensor 전체에 대해
평균을 내므로, vehicle type별 결과가 필요할 때는 원하는 channel만 먼저 선택해
이 함수를 호출하면 된다.
"""

# PyTorch tensor의 산술 연산을 사용하기 위해 torch를 불러온다.
import torch


def regression_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """prediction과 target의 전체 원소 기준 MAE와 RMSE를 Python float로 반환한다."""

    # 예측값에서 실제값을 빼 element-wise error tensor를 만든다.
    error = prediction - target
    # 절댓값 평균과 제곱오차 평균의 제곱근을 계산해 dictionary로 반환한다.
    return {
        # 모든 lane, channel, sample에 대한 mean absolute error를 계산한다.
        "mae": error.abs().mean().item(),
        # 모든 lane, channel, sample에 대한 root mean squared error를 계산한다.
        "rmse": error.square().mean().sqrt().item(),
    }
