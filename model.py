from models.model import InformerStack
import torch


if __name__ == '__main__':
    
    batch_size = 16
    
    enc_in = dec_in = 16
    c_out = 1
    seq_len = label_len = out_len = 12
    
    
    model = InformerStack(enc_in - 4,           # 인코더 입력 차원 수 (예: 피처 수 16)
                          dec_in - 4,           # 디코더 입력 차원 수 (예: 피처 수 16)
                          c_out,            # 출력 차원 수 (예: 예측할 온도 값의 차원 1)
                          seq_len,          # 인코더 입력 시퀀스 길이 (과거 데이터 길이)
                          label_len,        # 디코더 입력 시퀀스 길이
                          out_len,          # 예측할 출력 시퀀스 길이 (미래 예측 길이)
                          dropout=0.2,      # 드롭아웃 비율
                          )
    
    # 임의의 입력 데이터 생성 (배치 크기: batch_size)
    x_enc = torch.rand(batch_size, seq_len, enc_in)        # 인코더 입력 데이터
    x_mark_enc = x_enc[:,:,:4]

    # 모델 실행
    output = model(x_enc[:,:,4:], x_mark_enc, x_enc[:,:,4:], x_mark_enc)

    print("batch's data shape :",x_enc.shape)
    print("batch, seqlen, time :",x_mark_enc.shape)
    print("batch, seqlen, feature :",x_enc[:,:,4:].shape)
    # 출력 결과 확인
    print("Output shape:", output.shape)  # 예상 출력: [batch_size, out_len, c_out]
    