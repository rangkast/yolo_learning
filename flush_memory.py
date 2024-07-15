import torch

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")


# CUDA GPU 초기화
device = torch.device('cuda')

# GPU 메모리 캐시 비우기
torch.cuda.empty_cache()

# CUDA 가비지 컬렉션 트리거
torch.cuda.ipc_collect()
