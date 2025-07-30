import os
import argparse
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime
import copy

# 시각화 스타일 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Quantization backend 설정
torch.backends.quantized.engine = 'qnnpack'

# 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Quantization backend: {torch.backends.quantized.engine}")

# 하이퍼파라미터 - 더 긴 훈련을 위해 조정
ENV_NAME = "CartPole-v1"
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-3
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
TARGET_UPDATE = 100
TRAIN_EPISODES = 2000  # 300 -> 1000으로 증가
EVAL_EPISODES = 100

MODEL_DIR = "models"
NORMAL_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_normal.pth")
QAT_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_qat.pth")
RESULTS_DIR = "results"

# 추가: 더 세밀한 분석을 위한 설정
EXTENDED_TRAIN_EPISODES = 2000  # 확장 훈련용
DETAILED_EVAL_EPISODES = 500    # 더 정확한 평가용

# 🔍 1단계: QAT 문제 진단을 위한 디버깅 도구들
class QATDiagnostics:
    """QAT 관련 문제를 진단하는 도구"""
    
    @staticmethod
    def check_pytorch_version():
        """PyTorch 버전 확인"""
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"Quantization 지원: {hasattr(torch, 'quantization')}")
        
        # 권장 버전 확인
        version_parts = torch.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major >= 1 and minor >= 8:
            print("✅ QAT 지원 버전입니다.")
        else:
            print("⚠️  PyTorch 1.8+ 버전을 권장합니다.")
    
    @staticmethod
    def diagnose_qat_model(model, model_name="QAT Model"):
        """QAT 모델 상태 진단"""
        print(f"\n=== {model_name} 진단 ===")
        
        # 1. 모델 모드 확인
        print(f"훈련 모드: {model.training}")
        
        # 2. 파라미터 상태 확인
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"학습 가능한 파라미터: {trainable_params}/{total_params}")
        
        # 3. QAT 특화 확인
        has_fake_quant = False
        has_observer = False
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            if 'FakeQuantize' in module_type:
                has_fake_quant = True
                print(f"  FakeQuantize 발견: {name} -> {module_type}")
            elif 'Observer' in module_type:
                has_observer = True
                print(f"  Observer 발견: {name} -> {module_type}")
        
        print(f"FakeQuantize 모듈: {'있음' if has_fake_quant else '없음'}")
        print(f"Observer 모듈: {'있음' if has_observer else '없음'}")
        
        # 4. qconfig 확인
        if hasattr(model, 'qconfig') and model.qconfig:
            print(f"qconfig: {model.qconfig}")
        else:
            print("qconfig: 설정되지 않음")
        
        # 5. 각 파라미터의 gradient 상태
        print("파라미터 gradient 상태:")
        for name, param in model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}, "
                  f"grad={'있음' if param.grad is not None else '없음'}, "
                  f"device={param.device}")
    
    @staticmethod
    def test_forward_backward(model, input_shape, device='cpu'):
        """Forward/Backward 패스 테스트"""
        print(f"\n=== Forward/Backward 테스트 ===")
        
        model = model.to(device)
        model.train()
        
        # 테스트 입력
        x = torch.randn(2, *input_shape, device=device)
        target = torch.randn(2, 2, device=device)  # 2개 출력 가정
        
        # Forward 패스
        try:
            output = model(x)
            print(f"✅ Forward 성공: {output.shape}")
        except Exception as e:
            print(f"❌ Forward 실패: {e}")
            return False
        
        # Loss 계산
        loss = F.mse_loss(output, target)
        print(f"Loss: {loss.item():.6f}")
        
        # Backward 패스
        try:
            loss.backward()
            print("✅ Backward 성공")
            
            # Gradient 확인
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.norm() > 0:
                    grad_count += 1
                    print(f"  {name}: grad_norm={param.grad.norm().item():.8f}")
            
            if grad_count > 0:
                print(f"✅ {grad_count}개 파라미터에 gradient 존재")
                return True
            else:
                print("❌ Gradient가 없습니다!")
                return False
                
        except Exception as e:
            print(f"❌ Backward 실패: {e}")
            return False

# 일반 DQN 네트워크
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 올바른 Quantization-Aware Training을 위한 DQN
class QAT_DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QAT_DQN, self).__init__()
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 레이어들을 별도 모듈로 정의 (fuse_modules를 위해)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.relu1 = nn.ReLU(inplace=False)  # inplace=False가 중요!
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.quant(x)      # 입력 quantization
        x = self.fc1(x)        # fc1 가중치 fake quantization
        x = self.relu1(x)      # relu1 activation fake quantization
        x = self.fc2(x)        # fc2 가중치 fake quantization
        x = self.relu2(x)      # relu2 activation fake quantization
        x = self.fc3(x)        # fc3 가중치 fake quantization
        x = self.dequant(x)    # 출력 dequantization
        return x
        
    def fuse_model(self):
        """레이어 융합 (Linear + ReLU)"""
        # fc1과 relu1 융합
        torch.quantization.fuse_modules(self, [['fc1', 'relu1']], inplace=True)
        # fc2와 relu2 융합
        torch.quantization.fuse_modules(self, [['fc2', 'relu2']], inplace=True)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, use_qat=False):
        self.state_size = state_size
        self.action_size = action_size
        self.use_qat = use_qat
        
        # 네트워크 초기화
        if use_qat:
            self.q_network = QAT_DQN(state_size, action_size).to(device)
            self.target_network = QAT_DQN(state_size, action_size).to(device)
            
            # QAT 설정 - 올바른 순서로 적용
            # Step 1: 모델 융합 (Linear + ReLU)
            self.q_network.fuse_model()
            self.target_network.fuse_model()
            
            # Step 2: qconfig 설정 - 더 안정적인 설정 사용
            self.q_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            self.target_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            
            # Step 3: QAT 준비 - fake quantization 활성화
            torch.quantization.prepare_qat(self.q_network, inplace=True)
            torch.quantization.prepare_qat(self.target_network, inplace=True)
            
            print("QAT 설정 완료:")
            print(f"  - 융합된 모듈: fc1+relu1, fc2+relu2")
            print(f"  - qconfig: {self.q_network.qconfig}")
            print(f"  - fake quantization 활성화됨")
        else:
            self.q_network = DQN(state_size, action_size).to(device)
            self.target_network = DQN(state_size, action_size).to(device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)
        
        # 하이퍼파라미터
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.update_target_freq = 50
        self.step_count = 0
        
        # 훈련 기록용
        self.training_scores = []
        self.training_losses = []
        self.epsilon_history = []
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))
        
        # torch.FloatTensor(state): NumPy 배열을 PyTorch 텐서로 변환
        #.unsqueeze(0): 배치 차원 추가 (shape: [4] → [1, 4])
        #이유: 신경망은 배치 단위로 처리하기 때문
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # 모델이 quantized된 경우 또는 QAT 모델인 경우 CPU에서 실행
        is_quantized = (hasattr(self.q_network, '_modules') and 
                       any('quantized' in str(type(m)).lower() for m in self.q_network.modules()))
        
        if self.use_qat or is_quantized:
            # QAT 모델이나 quantized 모델은 CPU에서 실행
            state = state.to('cpu')
            self.q_network = self.q_network.to('cpu')
        else:
            # 일반 모델은 원래 디바이스에서 실행
            state = state.to(device)
            self.q_network = self.q_network.to(device)
            
        with torch.no_grad():
            q_values = self.q_network(state)
            # .cpu() : 결과를 CPU로 이동, data.numpy() : PyTorch 텐서를 NumPy 배열로 변환
        return np.argmax(q_values.cpu().data.numpy()) # 최대값의 인덱스 반환 (최적 행동)
     
    def replay(self):
        """향상된 replay 함수 - 학습 상태 모니터링"""
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        target_device = 'cpu' if self.use_qat else device
        
        states = torch.FloatTensor(states).to(target_device)
        actions = torch.LongTensor(actions).to(target_device)
        rewards = torch.FloatTensor(rewards).to(target_device)
        next_states = torch.FloatTensor(next_states).to(target_device)
        dones = torch.BoolTensor(dones).to(target_device)
        
        self.q_network = self.q_network.to(target_device)
        self.target_network = self.target_network.to(target_device)
        
        # Forward pass
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Loss 계산
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Gradient 계산 및 확인
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient norm 확인 (매 100 스텝마다)
        if self.step_count % 100 == 0:
            total_grad_norm = 0
            param_count = 0
            for name, param in self.q_network.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
                    param_count += 1
            
            #if param_count > 0:
            #    avg_grad_norm = total_grad_norm / param_count
            #    print(f"Step {self.step_count}: 평균 gradient norm = {avg_grad_norm:.6f}")
        
        # Gradient clipping (안정성 향상)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

    
    def benchmark_inference_speed(self, num_samples=1000):
        """추론 속도 벤치마크"""
        self.q_network.eval()
        
        # 테스트 입력 생성
        test_inputs = torch.randn(num_samples, self.state_size)
        
        # 모델이 quantized되었거나 QAT 모델인지 확인
        is_quantized = (hasattr(self.q_network, '_modules') and 
                       any('quantized' in str(type(m)).lower() for m in self.q_network.modules()))
        
        if self.use_qat or is_quantized:
            test_inputs = test_inputs.to('cpu')
            self.q_network = self.q_network.to('cpu')
            device_name = 'cpu'
        else:
            test_inputs = test_inputs.to(device)
            self.q_network = self.q_network.to(device)
            device_name = str(device)
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = self.q_network(test_inputs[:10])
        
        # 실제 측정
        if torch.cuda.is_available() and device_name != 'cpu':
            torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, num_samples, 32):
                batch = test_inputs[i:i+32]
                _ = self.q_network(batch)
        
        if torch.cuda.is_available() and device_name != 'cpu':
            torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_sample = (total_time / num_samples) * 1000
        throughput = num_samples / total_time
        
        return {
            'total_time': total_time,
            'avg_time_ms': avg_time_per_sample,
            'throughput': throughput,
            'is_quantized': is_quantized or self.use_qat,
            'device': device_name
        }
    
    def get_model_size_mb(self):
        """모델 크기 계산 (MB 단위)"""
        param_size = 0
        buffer_size = 0
        
        for param in self.q_network.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in self.q_network.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size = param_size + buffer_size
        return model_size / (1024 * 1024)
    
    def analyze_weight_distribution(self):
        """가중치 분포 분석"""
        weight_stats = {}
        
        for name, param in self.q_network.named_parameters():
            if 'weight' in name:
                weight_data = param.data.cpu().numpy().flatten()
                weight_stats[name] = {
                    'data': weight_data,
                    'min': float(np.min(weight_data)),
                    'max': float(np.max(weight_data)),
                    'mean': float(np.mean(weight_data)),
                    'std': float(np.std(weight_data)),
                    'unique_values': len(np.unique(weight_data)),
                    'dtype': str(param.dtype),
                    'shape': list(param.shape)
                }
        
        return weight_stats
    
    def save(self, path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if self.use_qat:
            # QAT 모델의 경우 eval 모드로 전환 후 convert
            self.q_network.eval()
            net_cpu = self.q_network.cpu()
            
            # 실제 quantized 모델로 변환
            try:
                net_quantized = torch.quantization.convert(net_cpu, inplace=False)
                print("QAT 모델이 성공적으로 quantized 모델로 변환되었습니다.")
                
                # 훈련 히스토리와 함께 저장
                torch.save({
                    'model_state_dict': net_quantized.state_dict(),
                    'training_scores': self.training_scores,
                    'training_losses': self.training_losses,
                    'epsilon_history': self.epsilon_history,
                    'use_qat': self.use_qat,
                    'is_quantized': True
                }, path)
                
                # 원래 디바이스로 복원 (훈련 계속하는 경우를 위해)
                self.q_network = self.q_network.to(device)
                self.q_network.train()
                
            except Exception as e:
                print(f"Quantization 변환 실패: {e}")
                print("QAT 모델을 float 상태로 저장합니다.")
                
                torch.save({
                    'model_state_dict': net_cpu.state_dict(),
                    'training_scores': self.training_scores,
                    'training_losses': self.training_losses,
                    'epsilon_history': self.epsilon_history,
                    'use_qat': self.use_qat,
                    'is_quantized': False
                }, path)
        else:
            # 일반 모델
            torch.save({
                'model_state_dict': self.q_network.state_dict(),
                'training_scores': self.training_scores,
                'training_losses': self.training_losses,
                'epsilon_history': self.epsilon_history,
                'use_qat': self.use_qat,
                'is_quantized': False
            }, path)
    
    
    def load_pretrained_weights(self, pretrained_state_dict):
        """사전 훈련된 가중치를 올바르게 전이"""
        print("\n🔄 가중치 전이 시작...")
        
        # 1. 임시 QAT 모델 생성 (융합 전)
        temp_qat = QAT_DQN(self.state_size, self.action_size).to('cpu')
        temp_qat.train()
        
        # 2. 가중치 복사 (융합 전에 수행)
        print("가중치 복사 중...")
        mapping = {
            'fc1.weight': 'fc1.weight',
            'fc1.bias': 'fc1.bias', 
            'fc2.weight': 'fc2.weight',
            'fc2.bias': 'fc2.bias',
            'fc3.weight': 'fc3.weight',
            'fc3.bias': 'fc3.bias'
        }
        
        for qat_key, normal_key in mapping.items():
            if normal_key in pretrained_state_dict:
                temp_qat.state_dict()[qat_key].copy_(pretrained_state_dict[normal_key])
                print(f"  {normal_key} -> {qat_key} ✅")
        
        # 3. 전이 검증 (융합 전 테스트)
        temp_qat.eval()
        test_input = torch.randn(1, self.state_size)
        with torch.no_grad():
            output_before = temp_qat(test_input)
        print(f"융합 전 출력: {output_before}")
        
        # 4. QAT 설정 적용
        temp_qat.train()
        temp_qat.fuse_model()
        temp_qat.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(temp_qat, inplace=True)
        
        # 5. QAT 설정 후 테스트
        temp_qat.eval()
        with torch.no_grad():
            output_after = temp_qat(test_input)
        print(f"QAT 설정 후 출력: {output_after}")
        print(f"출력 차이: {torch.norm(output_after - output_before).item():.6f}")
        
        # 6. 최종 모델 교체
        temp_qat.train()
        self.q_network = temp_qat
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 7. Optimizer 재생성 (중요!)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=5e-4)
        
        print("✅ 가중치 전이 완료!")
        
        # 8. 전이 후 진단
        QATDiagnostics.diagnose_qat_model(self.q_network, "가중치 전이 후")
    
    def set_qat_training_mode(self, phase="fine_tune"):
        """개선된 QAT 훈련 모드 설정"""
        if phase == "fine_tune":
            # Fine-tuning: 적절한 탐험과 작은 learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 5e-4  # 더 작은 LR
            self.epsilon = 0.3  # 적절한 탐험 유지
            self.epsilon_decay = 0.998  # 천천히 감소
            self.epsilon_min = 0.05  # 최소값도 조정
            print(f"QAT Fine-tuning 모드 설정: LR={5e-4}, ε={0.3}")
        
        elif phase == "normal":
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 1e-3
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            print("일반 훈련 모드로 복원")

    def warmup_replay_buffer(self, env, episodes=50):
        """Replay buffer 사전 채우기"""
        print(f"🔄 Replay buffer 워밍업 시작 ({episodes} 에피소드)...")
        
        # 임시로 높은 epsilon 설정 (다양한 경험 수집)
        original_epsilon = self.epsilon
        self.epsilon = 0.8
        
        for episode in range(episodes):
            obs = env.reset()
            state = obs[0] if isinstance(obs, tuple) else obs
            
            while True:
                action = self.act(state)
                step_result = env.step(action)
                
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                if done:
                    break
        
        self.epsilon = original_epsilon
        print(f"✅ Buffer 크기: {len(self.memory)}")

    
    def load(self, path):
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # 새로운 형식의 체크포인트인지 확인
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                is_quantized = checkpoint.get('is_quantized', False)
                
                if self.use_qat:
                    if is_quantized:
                        # 이미 quantized된 모델 로드
                        print("Quantized 모델을 로드합니다...")
                        net_fp = QAT_DQN(self.state_size, self.action_size)
                        net_fp.fuse_model()
                        net_fp.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                        torch.quantization.prepare_qat(net_fp, inplace=True)
                        net_q = torch.quantization.convert(net_fp.eval(), inplace=False)
                        net_q.load_state_dict(checkpoint['model_state_dict'])
                        self.q_network = net_q.to('cpu')
                    else:
                        # QAT 모델이지만 아직 quantized되지 않은 경우
                        print("QAT 모델을 로드합니다...")
                        self.q_network = QAT_DQN(self.state_size, self.action_size)
                        self.q_network.fuse_model()
                        self.q_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                        torch.quantization.prepare_qat(self.q_network, inplace=True)
                        self.q_network.load_state_dict(checkpoint['model_state_dict'])
                        self.q_network = self.q_network.to('cpu')
                else:
                    # 일반 모델
                    self.q_network = DQN(self.state_size, self.action_size)
                    self.q_network.load_state_dict(checkpoint['model_state_dict'])
                    self.q_network = self.q_network.to(device)
                
                # 훈련 히스토리 로드
                self.training_scores = checkpoint.get('training_scores', [])
                self.training_losses = checkpoint.get('training_losses', [])
                self.epsilon_history = checkpoint.get('epsilon_history', [])
                
            else:
                # 기존 형식 (state_dict만 저장된 경우)
                print(f"기존 형식의 모델 파일을 로드합니다: {path}")
                if self.use_qat:
                    self.q_network = QAT_DQN(self.state_size, self.action_size)
                    self.q_network.fuse_model()
                    self.q_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                    torch.quantization.prepare_qat(self.q_network, inplace=True)
                    # 기존 형식은 quantized되지 않은 상태로 가정
                    self.q_network.load_state_dict(checkpoint)
                    self.q_network = self.q_network.to('cpu')
                else:
                    self.q_network = DQN(self.state_size, self.action_size)
                    self.q_network.load_state_dict(checkpoint)
                    self.q_network = self.q_network.to(device)
                
                # 기존 형식에서는 훈련 히스토리가 없음
                self.training_scores = []
                self.training_losses = []
                self.epsilon_history = []
                print("기존 모델 파일에는 훈련 히스토리가 없습니다.")
            
            self.q_network.eval()
            
            # 타겟 네트워크도 같은 디바이스로 설정
            if hasattr(self, 'target_network'):
                if self.use_qat:
                    self.target_network = self.target_network.to('cpu')
                else:
                    self.target_network = self.target_network.to(device)
            
            print(f"모델이 성공적으로 로드되었습니다: {path}")
            print(f"모델 디바이스: {'CPU (QAT/Quantized)' if self.use_qat else device}")
            print(f"Quantized 상태: {checkpoint.get('is_quantized', 'Unknown')}")
            
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            print("모델 파일을 다시 훈련해주세요.")
            raise

def improved_transfer_learn_qat(normal_model_path, state_size, action_size, env, episodes=500):
    """개선된 QAT 전이 학습"""
    
    print("\n" + "="*60)
    print("🔄 개선된 QAT 전이 학습 시작")
    print("="*60)
    
    # 1. Normal 모델 로드 및 검증
    print("1. 사전 훈련된 Normal DQN 로드 중...")
    normal_agent = DQNAgent(state_size, action_size, use_qat=False)
    
    try:
        normal_agent.load(normal_model_path)
        normal_scores = evaluate_agent(normal_agent, env, episodes=50)
        normal_avg = np.mean(normal_scores)
        print(f"   📈 Normal 모델 평균 점수: {normal_avg:.2f}")
        
        if normal_avg < 200:
            print(f"   ⚠️  경고: Normal 모델 성능이 낮습니다 ({normal_avg:.2f})")
            print("   더 오래 훈련된 모델을 사용하는 것을 권장합니다.")
            
    except Exception as e:
        print(f"   ❌ Normal 모델 로드 실패: {e}")
        return None
    
    # 2. QAT 모델 생성 및 가중치 전이
    print("\n2. QAT 모델 생성 및 개선된 가중치 전이...")
    qat_agent = DQNAgent(state_size, action_size, use_qat=True)
    qat_agent.load_pretrained_weights(normal_agent.q_network.state_dict())
    
    # 3. Replay buffer 워밍업
    print("\n3. Replay buffer 워밍업...")
    qat_agent.warmup_replay_buffer(env, episodes=100)
    
    # 4. QAT Fine-tuning 설정
    print("\n4. QAT Fine-tuning 설정...")
    qat_agent.set_qat_training_mode("fine_tune")
    
    # 5. 초기 성능 확인 (워밍업 후)
    print("\n5. 워밍업 후 초기 성능 확인...")
    initial_scores = evaluate_agent(qat_agent, env, episodes=50)
    initial_avg = np.mean(initial_scores)
    print(f"   📈 워밍업 후 QAT 점수: {initial_avg:.2f}")
    print(f"   📊 성능 유지율: {(initial_avg/normal_avg)*100:.1f}%")
    
    # 6. 학습 추적을 위한 변수들
    print(f"\n6. QAT Fine-tuning 시작 ({episodes} 에피소드)...")
    
    # 가중치 변화 추적
    initial_weight = qat_agent.q_network.fc1.weight.data.clone()
    weight_changes = []
    performance_checks = []
    
    # 7. 실제 Fine-tuning (개선된 훈련 루프)
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0
        episode_losses = []
        
        while True:
            action = qat_agent.act(state)
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            qat_agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
            
            # 향상된 replay 사용
            loss = qat_agent.replay()
            if loss is not None:
                episode_losses.append(loss)
        
        scores_window.append(total_reward)
        scores.append(total_reward)
        qat_agent.training_scores.append(total_reward)
        qat_agent.epsilon_history.append(qat_agent.epsilon)
        
        if episode_losses:
            qat_agent.training_losses.append(np.mean(episode_losses))
        
        # 주기적 성능 및 가중치 변화 확인
        if (episode + 1) % 50 == 0:
            current_weight = qat_agent.q_network.fc1.weight.data.clone()
            weight_change = torch.norm(current_weight - initial_weight).item()
            weight_changes.append(weight_change)
            
            recent_avg = np.mean(list(scores_window)[-50:])
            performance_checks.append(recent_avg)
            
            print(f"Episode {episode+1:3d}: "
                  f"Score={total_reward:3.0f}, "
                  f"Avg50={recent_avg:5.1f}, "
                  f"WeightΔ={weight_change:.6f}, "
                  f"ε={qat_agent.epsilon:.3f}")
            
            # 학습 정체 확인
            if len(weight_changes) >= 4:
                recent_changes = weight_changes[-4:]
                if max(recent_changes) - min(recent_changes) < 1e-6:
                    print("   ⚠️  가중치 변화가 거의 없습니다. 학습이 정체되었을 수 있습니다.")
    
    # 8. 최종 성능 평가
    print("\n8. 최종 성능 평가...")
    final_scores = evaluate_agent(qat_agent, env, episodes=100)
    final_avg = np.mean(final_scores)
    
    final_weight = qat_agent.q_network.fc1.weight.data.clone()
    total_weight_change = torch.norm(final_weight - initial_weight).item()
    
    # 9. 결과 요약
    print("\n" + "="*60)
    print("📊 개선된 QAT 전이 학습 결과")
    print("="*60)
    print(f"🔵 Original Normal DQN:     {normal_avg:.2f} ± {np.std(normal_scores):.2f}")
    print(f"🟡 Initial QAT (워밍업후):   {initial_avg:.2f} ± {np.std(initial_scores):.2f}")
    print(f"🔴 Final QAT (파인튜닝후):   {final_avg:.2f} ± {np.std(final_scores):.2f}")
    print(f"📈 최종 성능 유지율:        {(final_avg/normal_avg)*100:.1f}%")
    print(f"🔧 총 가중치 변화량:        {total_weight_change:.6f}")
    print(f"📚 최종 Buffer 크기:        {len(qat_agent.memory)}")
    print(f"🎯 최종 Epsilon:           {qat_agent.epsilon:.4f}")
    
    # 학습 성공 여부 판단
    learning_success = (
        total_weight_change > 1e-4 and  # 가중치가 충분히 변했는지
        final_avg > initial_avg * 0.9   # 성능이 유지되거나 개선되었는지
    )
    
    if learning_success:
        print("✅ QAT Fine-tuning이 성공적으로 완료되었습니다!")
    else:
        print("⚠️  QAT Fine-tuning에 문제가 있을 수 있습니다.")
        print(f"   가중치 변화: {'충분함' if total_weight_change > 1e-4 else '부족함'}")
        print(f"   성능 유지: {'성공' if final_avg > initial_avg * 0.9 else '실패'}")
    
    return qat_agent

def plot_transfer_learning_results(normal_score, initial_qat_score, final_qat_score, training_scores):
    """전이 학습 결과 시각화"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('QAT Transfer Learning Results', fontsize=16, fontweight='bold')
    
    # 1. 성능 비교 바 차트
    stages = ['Normal\n(Original)', 'QAT\n(Initial)', 'QAT\n(Fine-tuned)']
    scores = [normal_score, initial_qat_score, final_qat_score]
    colors = ['blue', 'orange', 'red']
    
    bars = ax1.bar(stages, scores, color=colors, alpha=0.7)
    ax1.set_title('Performance Comparison')
    ax1.set_ylabel('Average Score')
    ax1.grid(True, alpha=0.3)
    
    # 값 표시
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 성능 유지율 표시
    retention_rate = (final_qat_score / normal_score) * 100
    ax1.text(0.5, 0.95, f'Performance Retention: {retention_rate:.1f}%', 
             transform=ax1.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    # 2. Fine-tuning 학습 곡선
    if training_scores:
        episodes = range(len(training_scores))
        ax2.plot(episodes, training_scores, color='red', alpha=0.7, label='QAT Fine-tuning')
        
        # 이동평균 추가
        if len(training_scores) >= 50:
            window_size = 50
            moving_avg = np.convolve(training_scores, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(range(window_size-1, len(training_scores)), moving_avg, 
                    color='darkred', linewidth=2, label='Moving Average (50)')
        
        # 목표선 (Normal 성능)
        ax2.axhline(y=normal_score, color='blue', linestyle='--', alpha=0.8, 
                   label=f'Target (Normal: {normal_score:.1f})')
        
        ax2.set_title('QAT Fine-tuning Progress')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'qat_transfer_learning.png'), dpi=300, bbox_inches='tight')
    plt.show()
    try:
        checkpoint = torch.load(path, map_location='cpu')
        
        # 새로운 형식의 체크포인트인지 확인
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            is_quantized = checkpoint.get('is_quantized', False)
            
            if self.use_qat:
                if is_quantized:
                    # 이미 quantized된 모델 로드
                    print("Quantized 모델을 로드합니다...")
                    net_fp = QAT_DQN(self.state_size, self.action_size)
                    net_fp.fuse_model()
                    net_fp.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                    torch.quantization.prepare_qat(net_fp, inplace=True)
                    net_q = torch.quantization.convert(net_fp.eval(), inplace=False)
                    net_q.load_state_dict(checkpoint['model_state_dict'])
                    self.q_network = net_q.to('cpu')
                else:
                    # QAT 모델이지만 아직 quantized되지 않은 경우
                    print("QAT 모델을 로드합니다...")
                    self.q_network = QAT_DQN(self.state_size, self.action_size)
                    self.q_network.fuse_model()
                    self.q_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                    torch.quantization.prepare_qat(self.q_network, inplace=True)
                    self.q_network.load_state_dict(checkpoint['model_state_dict'])
                    self.q_network = self.q_network.to('cpu')
            else:
                # 일반 모델
                self.q_network = DQN(self.state_size, self.action_size)
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
                self.q_network = self.q_network.to(device)
            
            # 훈련 히스토리 로드
            self.training_scores = checkpoint.get('training_scores', [])
            self.training_losses = checkpoint.get('training_losses', [])
            self.epsilon_history = checkpoint.get('epsilon_history', [])
            
        else:
            # 기존 형식 (state_dict만 저장된 경우)
            print(f"기존 형식의 모델 파일을 로드합니다: {path}")
            if self.use_qat:
                self.q_network = QAT_DQN(self.state_size, self.action_size)
                self.q_network.fuse_model()
                self.q_network.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                torch.quantization.prepare_qat(self.q_network, inplace=True)
                # 기존 형식은 quantized되지 않은 상태로 가정
                self.q_network.load_state_dict(checkpoint)
                self.q_network = self.q_network.to('cpu')
            else:
                self.q_network = DQN(self.state_size, self.action_size)
                self.q_network.load_state_dict(checkpoint)
                self.q_network = self.q_network.to(device)
            
            # 기존 형식에서는 훈련 히스토리가 없음
            self.training_scores = []
            self.training_losses = []
            self.epsilon_history = []
            print("기존 모델 파일에는 훈련 히스토리가 없습니다.")
        
        self.q_network.eval()
        
        # 타겟 네트워크도 같은 디바이스로 설정
        if hasattr(self, 'target_network'):
            if self.use_qat:
                self.target_network = self.target_network.to('cpu')
            else:
                self.target_network = self.target_network.to(device)
        
        print(f"모델이 성공적으로 로드되었습니다: {path}")
        print(f"모델 디바이스: {'CPU (QAT/Quantized)' if self.use_qat else device}")
        print(f"Quantized 상태: {checkpoint.get('is_quantized', 'Unknown')}")
        
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("모델 파일을 다시 훈련해주세요.")
        raise

def validate_qat_learning(agent, episodes=10):
    """QAT 모델의 학습 상태 검증"""
    print("\n=== QAT 학습 상태 검증 ===")
    
    # 1. 가중치 분석
    weight_stats = agent.analyze_weight_distribution()
    for layer_name, stats in weight_stats.items():
        print(f"{layer_name}: unique_values={stats['unique_values']}, std={stats['std']:.6f}")
    
    # 2. Gradient 흐름 확인
    if len(agent.memory) >= agent.batch_size:
        print("Gradient 흐름 테스트...")
        for i in range(3):  # 3번 테스트
            loss = agent.replay()
            if loss:
                print(f"  Test {i+1}: loss={loss:.6f}")
    
    # 3. 모델 출력 일관성 확인
    agent.q_network.eval()
    test_input = torch.randn(5, agent.state_size).to('cpu')
    
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            output = agent.q_network(test_input)
            outputs.append(output)
    
    # 출력 일관성 확인
    consistency = all(torch.allclose(outputs[0], out, atol=1e-6) for out in outputs[1:])
    print(f"출력 일관성: {'일관됨' if consistency else '불일치 (정상적인 quantization 효과)'}")
    
    agent.q_network.train()
    
    return {
        'weight_stats': weight_stats,
        'output_consistency': consistency
    }


# 에이전트 학습 함수
def train_agent(agent, env, episodes=300, agent_name="Agent"):
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            state = obs[0]
        else:
            state = obs
        total_reward = 0
        episode_losses = []
        
        while True:
            action = agent.act(state)
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done:
                break
            
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
        
        print(f"Episode {episode} 개별 점수: {total_reward}")

        scores_window.append(total_reward)
        scores.append(total_reward)
        agent.training_scores.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        if episode_losses:
            agent.training_losses.append(np.mean(episode_losses))
        
        if episode % 10 == 0:  # 더 자주 출력
            recent_scores = list(scores_window)[-10:]  # 최근 10개
            print(f"최근 10개 에피소드: {recent_scores}")
            print(f"평균: {np.mean(scores_window):.2f}")
    
    return scores

# 에이전트 평가 함수
def evaluate_agent(agent, env, episodes=100):
    scores = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for _ in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state) # 선택된 행동 (0=왼쪽, 1=오른쪽)
            step = env.step(action) # step : 환경의 반응 (다음 상태, 보상, 종료 여부 등)
            if len(step) == 5:
                next_state, reward, term, trunc, _ = step
                done = term or trunc
            else:
                next_state, reward, done, _ = step
            state = next_state
            total_reward += reward
        
        scores.append(total_reward)
    
    agent.epsilon = original_epsilon
    return scores

# 실시간 시뮬레이션 함수들
def run_live_simulation(agent, env, agent_name="Agent", episodes=5, delay=0.05):
    """실시간 시뮬레이션 실행"""
    print(f"\n=== {agent_name} 실시간 시뮬레이션 ===")
    
    # 시뮬레이션 데이터 저장용
    simulation_data = {
        'positions': [],
        'angles': [],
        'actions': [],
        'rewards': [],
        'episode_scores': []
    }
    
    agent.epsilon = 0  # 탐험 비활성화
    
    for episode in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0
        step_count = 0
        
        episode_positions = []
        episode_angles = []
        episode_actions = []
        episode_rewards = []
        
        print(f"\n에피소드 {episode + 1} 시작...")
        
        while True:
            # 환경 렌더링
            if hasattr(env, 'render') and delay > 0:
                env.render()
            time.sleep(delay)
            
            # 상태 정보 저장
            cart_pos = state[0]
            pole_angle = state[2]
            episode_positions.append(cart_pos)
            episode_angles.append(pole_angle)
            
            # 행동 선택
            action = agent.act(state)
            episode_actions.append(action)
            
            # 환경 스텝
            step_result = env.step(action)
            if len(step_result) == 5:  # 최신 gym (terminated, truncated 분리)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # 구버전 gym
                next_state, reward, done, info = step_result
            
            episode_rewards.append(reward)
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 에피소드 데이터 저장
        simulation_data['positions'].append(episode_positions)
        simulation_data['angles'].append(episode_angles)
        simulation_data['actions'].append(episode_actions)
        simulation_data['rewards'].append(episode_rewards)
        simulation_data['episode_scores'].append(total_reward)
        
        print(f"에피소드 {episode + 1} 완료 - 점수: {total_reward}, 스텝: {step_count}")
    
    print(f"\n{agent_name} 평균 점수: {np.mean(simulation_data['episode_scores']):.2f}")
    return simulation_data

def compare_live_simulations(normal_agent, qat_agent, env, episodes=3):
    """두 에이전트의 실시간 시뮬레이션 비교"""
    print("\n=== 실시간 시뮬레이션 비교 ===")
    
    # Normal 에이전트 시뮬레이션
    print("\n1. Normal DQN 시뮬레이션 실행 중...")
    normal_data = run_live_simulation(normal_agent, env, "Normal DQN", episodes)
    
    input("\nQAT DQN 시뮬레이션을 시작하려면 Enter를 누르세요...")
    
    # QAT 에이전트 시뮬레이션
    print("\n2. QAT DQN 시뮬레이션 실행 중...")
    qat_data = run_live_simulation(qat_agent, env, "QAT DQN", episodes)
    
    # 결과 비교 시각화
    plot_simulation_comparison(normal_data, qat_data)
    
    return normal_data, qat_data

def plot_simulation_comparison(normal_data, qat_data):
    """시뮬레이션 결과 비교 시각화"""
    # results 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Live Simulation Comparison', fontsize=16, fontweight='bold')
    
    # 1. 평균 성능 비교
    normal_scores = normal_data['episode_scores']
    qat_scores = qat_data['episode_scores']
    
    episodes = range(1, len(normal_scores) + 1)
    width = 0.35
    x = np.arange(len(episodes))
    
    axes[0,0].bar(x - width/2, normal_scores, width, label='Normal DQN', color='blue', alpha=0.7)
    axes[0,0].bar(x + width/2, qat_scores, width, label='QAT DQN', color='red', alpha=0.7)
    axes[0,0].set_title('Episode Scores Comparison')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([f'Ep{i}' for i in episodes])
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 카트 위치 변화 (첫 번째 에피소드)
    if normal_data['positions'] and qat_data['positions']:
        normal_pos = normal_data['positions'][0]
        qat_pos = qat_data['positions'][0]
        
        steps_normal = range(len(normal_pos))
        steps_qat = range(len(qat_pos))
        
        axes[0,1].plot(steps_normal, normal_pos, label='Normal DQN', color='blue', alpha=0.7)
        axes[0,1].plot(steps_qat, qat_pos, label='QAT DQN', color='red', alpha=0.7)
        axes[0,1].set_title('Cart Position Over Time (Episode 1)')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Cart Position')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axhline(y=2.4, color='orange', linestyle='--', alpha=0.5, label='Boundary')
        axes[0,1].axhline(y=-2.4, color='orange', linestyle='--', alpha=0.5)
    
    # 3. 폴 각도 변화 (첫 번째 에피소드)
    if normal_data['angles'] and qat_data['angles']:
        normal_angles = normal_data['angles'][0]
        qat_angles = qat_data['angles'][0]
        
        steps_normal = range(len(normal_angles))
        steps_qat = range(len(qat_angles))
        
        axes[1,0].plot(steps_normal, np.degrees(normal_angles), label='Normal DQN', color='blue', alpha=0.7)
        axes[1,0].plot(steps_qat, np.degrees(qat_angles), label='QAT DQN', color='red', alpha=0.7)
        axes[1,0].set_title('Pole Angle Over Time (Episode 1)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Pole Angle (degrees)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='Boundary')
        axes[1,0].axhline(y=-12, color='orange', linestyle='--', alpha=0.5)
    
    # 4. 행동 분포 비교
    all_normal_actions = []
    all_qat_actions = []
    
    for episode_actions in normal_data['actions']:
        all_normal_actions.extend(episode_actions)
    for episode_actions in qat_data['actions']:
        all_qat_actions.extend(episode_actions)
    
    action_labels = ['Left', 'Right']
    normal_action_counts = [all_normal_actions.count(0), all_normal_actions.count(1)]
    qat_action_counts = [all_qat_actions.count(0), all_qat_actions.count(1)]
    
    x = np.arange(len(action_labels))
    width = 0.35
    
    axes[1,1].bar(x - width/2, normal_action_counts, width, label='Normal DQN', color='blue', alpha=0.7)
    axes[1,1].bar(x + width/2, qat_action_counts, width, label='QAT DQN', color='red', alpha=0.7)
    axes[1,1].set_title('Action Distribution')
    axes[1,1].set_xlabel('Action')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(action_labels)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'live_simulation_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_animated_comparison(normal_data, qat_data):
    """애니메이션 비교 생성"""
    import matplotlib.animation as animation
    
    # results 디렉토리 생성
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not normal_data['positions'] or not qat_data['positions']:
        print("시뮬레이션 데이터가 없습니다.")
        return
    
    # 첫 번째 에피소드 데이터 사용
    normal_pos = normal_data['positions'][0]
    normal_angles = normal_data['angles'][0]
    qat_pos = qat_data['positions'][0]
    qat_angles = qat_data['angles'][0]
    
    # 더 긴 에피소드에 맞춰 조정
    max_steps = max(len(normal_pos), len(qat_pos))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Real-time CartPole Comparison Animation', fontsize=16, fontweight='bold')
    
    # Normal DQN 서브플롯
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.5, 2)
    ax1.set_title('Normal DQN')
    ax1.set_xlabel('Position')
    ax1.grid(True, alpha=0.3)
    
    # QAT DQN 서브플롯
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-0.5, 2)
    ax2.set_title('QAT DQN')
    ax2.set_xlabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # 그래픽 요소 초기화
    from matplotlib.patches import Rectangle
    cart1 = Rectangle((-0.25, 0), 0.5, 0.3, fc='blue', alpha=0.7)
    pole1, = ax1.plot([], [], 'b-', linewidth=8)
    ax1.add_patch(cart1)
    
    cart2 = Rectangle((-0.25, 0), 0.5, 0.3, fc='red', alpha=0.7)
    pole2, = ax2.plot([], [], 'r-', linewidth=8)
    ax2.add_patch(cart2)
    
    # 점수 텍스트
    score_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    score_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Normal DQN 업데이트
        if frame < len(normal_pos):
            cart_x1 = normal_pos[frame]
            pole_angle1 = normal_angles[frame]
            
            # 카트 위치 업데이트
            cart1.set_x(cart_x1 - 0.25)
            
            # 폴 위치 계산 (길이 1.0 가정)
            pole_x1 = [cart_x1, cart_x1 + np.sin(pole_angle1)]
            pole_y1 = [0.15, 0.15 + np.cos(pole_angle1)]
            pole1.set_data(pole_x1, pole_y1)
            
            score_text1.set_text(f'Step: {frame + 1}\nScore: {frame + 1}')
        
        # QAT DQN 업데이트
        if frame < len(qat_pos):
            cart_x2 = qat_pos[frame]
            pole_angle2 = qat_angles[frame]
            
            # 카트 위치 업데이트
            cart2.set_x(cart_x2 - 0.25)
            
            # 폴 위치 계산
            pole_x2 = [cart_x2, cart_x2 + np.sin(pole_angle2)]
            pole_y2 = [0.15, 0.15 + np.cos(pole_angle2)]
            pole2.set_data(pole_x2, pole_y2)
            
            score_text2.set_text(f'Step: {frame + 1}\nScore: {frame + 1}')
        
        return cart1, pole1, cart2, pole2, score_text1, score_text2
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                 interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    
    # GIF로 저장 (선택사항)
    try:
        anim.save(os.path.join(RESULTS_DIR, 'cartpole_comparison.gif'), 
                 writer='pillow', fps=10)
        print(f"애니메이션이 '{RESULTS_DIR}/cartpole_comparison.gif'에 저장되었습니다.")
    except:
        print("GIF 저장에 실패했습니다. pillow 패키지가 필요합니다.")
    
    plt.show()
    return anim

# 시각화 함수들
class ModelComparator:
    def __init__(self, normal_agent, qat_agent):
        self.normal_agent = normal_agent
        self.qat_agent = qat_agent
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def plot_training_comparison(self):
        """훈련 과정 비교 시각화"""
        # results 디렉토리 생성
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress Comparison: Normal DQN vs QAT DQN', fontsize=16, fontweight='bold')
        
        # 1. 스코어 비교
        if self.normal_agent.training_scores and self.qat_agent.training_scores:
            episodes_normal = range(len(self.normal_agent.training_scores))
            episodes_qat = range(len(self.qat_agent.training_scores))
            
            axes[0,0].plot(episodes_normal, self.normal_agent.training_scores, 
                          label='Normal DQN', alpha=0.7, color='blue')
            axes[0,0].plot(episodes_qat, self.qat_agent.training_scores, 
                          label='QAT DQN', alpha=0.7, color='red')
            
            # 이동평균 추가
            window_size = 50
            if len(self.normal_agent.training_scores) >= window_size:
                normal_ma = np.convolve(self.normal_agent.training_scores, 
                                      np.ones(window_size)/window_size, mode='valid')
                axes[0,0].plot(range(window_size-1, len(self.normal_agent.training_scores)), 
                              normal_ma, label='Normal MA', color='darkblue', linewidth=2)
            
            if len(self.qat_agent.training_scores) >= window_size:
                qat_ma = np.convolve(self.qat_agent.training_scores, 
                                   np.ones(window_size)/window_size, mode='valid')
                axes[0,0].plot(range(window_size-1, len(self.qat_agent.training_scores)), 
                              qat_ma, label='QAT MA', color='darkred', linewidth=2)
            
            axes[0,0].set_title('Training Scores')
            axes[0,0].set_xlabel('Episode')
            axes[0,0].set_ylabel('Score')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. 손실 함수 비교
        if self.normal_agent.training_losses and self.qat_agent.training_losses:
            axes[0,1].plot(self.normal_agent.training_losses, 
                          label='Normal DQN', alpha=0.7, color='blue')
            axes[0,1].plot(self.qat_agent.training_losses, 
                          label='QAT DQN', alpha=0.7, color='red')
            axes[0,1].set_title('Training Loss')
            axes[0,1].set_xlabel('Episode')
            axes[0,1].set_ylabel('Loss')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_yscale('log')
        
        # 3. Epsilon 감소 비교
        if self.normal_agent.epsilon_history and self.qat_agent.epsilon_history:
            axes[1,0].plot(self.normal_agent.epsilon_history, 
                          label='Normal DQN', color='blue')
            axes[1,0].plot(self.qat_agent.epsilon_history, 
                          label='QAT DQN', color='red')
            axes[1,0].set_title('Epsilon Decay')
            axes[1,0].set_xlabel('Episode')
            axes[1,0].set_ylabel('Epsilon')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 최종 성능 통계
        normal_final_scores = self.normal_agent.training_scores[-100:] if len(self.normal_agent.training_scores) >= 100 else self.normal_agent.training_scores
        qat_final_scores = self.qat_agent.training_scores[-100:] if len(self.qat_agent.training_scores) >= 100 else self.qat_agent.training_scores
        
        if normal_final_scores and qat_final_scores:
            performance_data = [normal_final_scores, qat_final_scores]
            labels = ['Normal DQN', 'QAT DQN']
            
            box_plot = axes[1,1].boxplot(performance_data, labels=labels, patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            box_plot['boxes'][1].set_facecolor('lightcoral')
            axes[1,1].set_title('Final Performance Distribution\n(Last 100 episodes)')
            axes[1,1].set_ylabel('Score')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'training_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weight_distribution_comparison(self):
        """가중치 분포 비교"""
        # results 디렉토리 생성
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        normal_weights = self.normal_agent.analyze_weight_distribution()
        qat_weights = self.qat_agent.analyze_weight_distribution()
        
        # 공통 레이어 찾기
        common_layers = set(normal_weights.keys()) & set(qat_weights.keys())
        
        if not common_layers:
            print("공통 레이어를 찾을 수 없습니다.")
            return
        
        n_layers = len(common_layers)
        fig, axes = plt.subplots(n_layers, 2, figsize=(15, 4*n_layers))
        fig.suptitle('Weight Distribution Comparison', fontsize=16, fontweight='bold')
        
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, layer_name in enumerate(sorted(common_layers)):
            # Normal 모델 히스토그램
            normal_data = normal_weights[layer_name]['data']
            axes[i,0].hist(normal_data, bins=50, alpha=0.7, color='blue', density=True)
            axes[i,0].set_title(f'Normal DQN - {layer_name}')
            axes[i,0].set_xlabel('Weight Value')
            axes[i,0].set_ylabel('Density')
            axes[i,0].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            stats_text = f'Mean: {normal_weights[layer_name]["mean"]:.4f}\n' \
                        f'Std: {normal_weights[layer_name]["std"]:.4f}\n' \
                        f'Unique: {normal_weights[layer_name]["unique_values"]}'
            axes[i,0].text(0.02, 0.98, stats_text, transform=axes[i,0].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # QAT 모델 히스토그램
            qat_data = qat_weights[layer_name]['data']
            axes[i,1].hist(qat_data, bins=50, alpha=0.7, color='red', density=True)
            axes[i,1].set_title(f'QAT DQN - {layer_name}')
            axes[i,1].set_xlabel('Weight Value')
            axes[i,1].set_ylabel('Density')
            axes[i,1].grid(True, alpha=0.3)
            
            # 통계 정보 추가
            stats_text = f'Mean: {qat_weights[layer_name]["mean"]:.4f}\n' \
                        f'Std: {qat_weights[layer_name]["std"]:.4f}\n' \
                        f'Unique: {qat_weights[layer_name]["unique_values"]}'
            axes[i,1].text(0.02, 0.98, stats_text, transform=axes[i,1].transAxes, 
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'weight_distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, eval_env):
        """성능 메트릭 비교"""
        print("성능 평가 중...")
        
        # results 디렉토리 생성
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # 평가 수행
        normal_eval_scores = evaluate_agent(self.normal_agent, eval_env, episodes=100)
        qat_eval_scores = evaluate_agent(self.qat_agent, eval_env, episodes=100)
        
        # 추론 속도 벤치마크
        normal_speed = self.normal_agent.benchmark_inference_speed(1000)
        qat_speed = self.qat_agent.benchmark_inference_speed(1000)
        
        # 모델 크기
        normal_size = self.normal_agent.get_model_size_mb()
        qat_size = self.qat_agent.get_model_size_mb()
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # 1. 평가 점수 분포
        axes[0,0].boxplot([normal_eval_scores, qat_eval_scores], 
                         labels=['Normal DQN', 'QAT DQN'], patch_artist=True)
        axes[0,0].set_title('Evaluation Scores Distribution')
        axes[0,0].set_ylabel('Score')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 추론 속도 비교
        models = ['Normal DQN', 'QAT DQN']
        throughputs = [normal_speed['throughput'], qat_speed['throughput']]
        colors = ['blue', 'red']
        
        bars = axes[0,1].bar(models, throughputs, color=colors, alpha=0.7)
        axes[0,1].set_title('Inference Throughput')
        axes[0,1].set_ylabel('Samples/sec')
        axes[0,1].grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, throughputs):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                          f'{value:.1f}', ha='center', va='bottom')
        
        # 3. 모델 크기 비교
        sizes = [normal_size, qat_size]
        bars = axes[1,0].bar(models, sizes, color=colors, alpha=0.7)
        axes[1,0].set_title('Model Size')
        axes[1,0].set_ylabel('Size (MB)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 값 표시
        for bar, value in zip(bars, sizes):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 종합 메트릭 비교
        categories = ['Performance\n(Score)', 'Speed\n(Throughput)', 'Efficiency\n(1/Size)']
        
        # 정규화 (0-1 스케일)
        normal_perf = np.mean(normal_eval_scores) / 500  # CartPole 최대 점수 기준
        qat_perf = np.mean(qat_eval_scores) / 500
        
        normal_speed_norm = normal_speed['throughput'] / max(normal_speed['throughput'], qat_speed['throughput'])
        qat_speed_norm = qat_speed['throughput'] / max(normal_speed['throughput'], qat_speed['throughput'])
        
        normal_eff = (1/normal_size) / max(1/normal_size, 1/qat_size)
        qat_eff = (1/qat_size) / max(1/normal_size, 1/qat_size)
        
        normal_values = [normal_perf, normal_speed_norm, normal_eff]
        qat_values = [qat_perf, qat_speed_norm, qat_eff]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1,1].bar(x - width/2, normal_values, width, label='Normal DQN', color='blue', alpha=0.7)
        axes[1,1].bar(x + width/2, qat_values, width, label='QAT DQN', color='red', alpha=0.7)
        
        axes[1,1].set_title('Normalized Metrics Comparison')
        axes[1,1].set_ylabel('Normalized Score (0-1)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(categories)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 수치 결과 출력
        print("\n=== 성능 비교 결과 ===")
        print(f"Normal DQN - 평균 점수: {np.mean(normal_eval_scores):.2f} ± {np.std(normal_eval_scores):.2f}")
        print(f"QAT DQN - 평균 점수: {np.mean(qat_eval_scores):.2f} ± {np.std(qat_eval_scores):.2f}")
        print(f"Normal DQN - 추론 속도: {normal_speed['throughput']:.1f} samples/sec")
        print(f"QAT DQN - 추론 속도: {qat_speed['throughput']:.1f} samples/sec")
        print(f"Normal DQN - 모델 크기: {normal_size:.3f} MB")
        print(f"QAT DQN - 모델 크기: {qat_size:.3f} MB")
        print(f"크기 압축률: {(normal_size/qat_size):.2f}x")
    
    def generate_comprehensive_report(self, eval_env):
        """종합 비교 리포트 생성"""
        print("종합 비교 리포트 생성 중...")
        
        # 모든 시각화 생성
        self.plot_training_comparison()
        self.plot_weight_distribution_comparison()
        self.plot_performance_metrics(eval_env)
        
        print(f"\n모든 비교 차트가 '{RESULTS_DIR}' 폴더에 저장되었습니다.")
        print("- training_comparison.png: 훈련 과정 비교")
        print("- weight_distribution_comparison.png: 가중치 분포 비교")
        print("- performance_metrics.png: 성능 메트릭 비교")

# 추가 유틸리티 함수들
def print_model_info(agent, agent_name):
    """모델 정보 출력"""
    print(f"\n=== {agent_name} 모델 정보 ===")
    print(f"모델 타입: {'QAT' if agent.use_qat else 'Normal'}")
    print(f"모델 크기: {agent.get_model_size_mb():.3f} MB")
    
    # 모델 구조 출력
    print("모델 구조:")
    for name, module in agent.q_network.named_modules():
        if len(list(module.children())) == 0:  # leaf 모듈만
            print(f"  {name}: {module}")
    
    # 가중치 통계
    weight_stats = agent.analyze_weight_distribution()
    print("가중치 통계:")
    for layer_name, stats in weight_stats.items():
        print(f"  {layer_name}: "
              f"min={stats['min']:.4f}, max={stats['max']:.4f}, "
              f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
              f"unique_values={stats['unique_values']}")

def print_usage_examples():
    """사용 예시 출력"""
    print("\n" + "="*60)
    print("🚀 QAT vs Normal DQN 비교 분석 도구")
    print("="*60)
    print()
    print("1. 처음 사용하는 경우:")
    print("   python test2.py --mode train_normal")
    print("   python test2.py --mode transfer_qat --episodes 1000")
    print("   python test2.py --mode compare")
    print()
    print("2. 🌟 QAT 전이 학습 (추천):")
    print("   python test2.py --mode transfer_qat --episodes 1000")
    print()
    print("3. 실시간 시뮬레이션 보기:")
    print("   python test2.py --mode simulate")
    print()
    print("4. 애니메이션 생성:")
    print("   python test2.py --mode animate")
    print()
    print("5. 기존 방식 (참고용):")
    print("   python test2.py --mode train_both")
    print()
    print("📁 결과 파일 위치:")
    print(f"   모델: {MODEL_DIR}/")
    print(f"   시각화: {RESULTS_DIR}/")
    print()
    print("💡 권장 순서:")
    print("   1) train_normal (2000+ episodes)")
    print("   2) transfer_qat (1000 episodes)")  
    print("   3) compare (전체 분석)")
    print("="*60)

# 메인 실행 부분
if __name__ == '__main__':
    # 인수가 없으면 도움말 출력
    if len(sys.argv) == 1:
        print_usage_examples()
        parser = argparse.ArgumentParser(description='QAT vs Normal DQN 비교 분석 도구')
        parser.add_argument('--mode', choices=['train_normal', 'train_qat', 'train_both', 'compare', 'simulate', 'animate', 'transfer_qat'], 
                          required=True, help='실행 모드 선택')
        parser.print_help()
        sys.exit(0)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train_normal', 'train_qat', 'train_both', 'compare', 'simulate', 'animate', 'transfer_qat'], required=True)
    args = parser.parse_args()

    # 환경 생성
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    if args.mode == 'train_normal':
        print("\n=== 일반 DQN 훈련 ===")
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        train_agent(normal_agent, env, episodes=TRAIN_EPISODES, agent_name="Normal DQN")
        normal_agent.save(NORMAL_MODEL_PATH)
        print(f"Normal DQN 모델이 {NORMAL_MODEL_PATH}에 저장되었습니다.")
        
    elif args.mode == 'train_qat':
        print("\n=== QAT DQN 훈련 ===")
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        train_agent(qat_agent, env, episodes=TRAIN_EPISODES, agent_name="QAT DQN")
        qat_agent.save(QAT_MODEL_PATH)
        print(f"QAT DQN 모델이 {QAT_MODEL_PATH}에 저장되었습니다.")
        
    elif args.mode == 'transfer_qat':
        print("\n=== QAT 전이 학습 ===")
        
        # 사전 훈련된 Normal 모델 확인
        if not os.path.exists(NORMAL_MODEL_PATH):
            print("❌ 사전 훈련된 Normal 모델을 찾을 수 없습니다.")
            print("먼저 다음 명령으로 Normal 모델을 훈련하세요:")
            print("python test2.py --mode train_normal")
            sys.exit(1)
        
        # QAT 전이 학습 실행
        qat_agent = improved_transfer_learn_qat(
            NORMAL_MODEL_PATH, state_size, action_size, env, episodes=2000
        )
        
        if qat_agent:
            # 학습 검증
            validate_qat_learning(qat_agent)
            
            # 모델 저장
            qat_agent.save(QAT_MODEL_PATH)
            print(f"\n✅ 개선된 QAT 모델이 저장되었습니다: {QAT_MODEL_PATH}")
        
    elif args.mode == 'train_both':
        print("\n=== 두 모델 모두 훈련 ===")
        
        # Normal DQN 훈련
        print("\n1. 일반 DQN 훈련 시작...")
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        train_agent(normal_agent, env, episodes=TRAIN_EPISODES, agent_name="Normal DQN")
        normal_agent.save(NORMAL_MODEL_PATH)
        
        # QAT DQN 훈련
        print("\n2. QAT DQN 훈련 시작...")
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        train_agent(qat_agent, env, episodes=TRAIN_EPISODES, agent_name="QAT DQN")
        qat_agent.save(QAT_MODEL_PATH)
        
        print("\n두 모델 훈련이 완료되었습니다!")
        
    elif args.mode == 'compare':
        print("\n=== 모델 비교 분석 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다.")
            print("다음 중 하나를 먼저 실행하세요:")
            print("  python test2.py --mode train_both")
            print("  python test2.py --mode transfer_qat")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 비교 분석 실행
        comparator = ModelComparator(normal_agent, qat_agent)
        comparator.generate_comprehensive_report(env)
        
    elif args.mode == 'simulate':
        print("\n=== 실시간 시뮬레이션 모드 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다.")
            print("먼저 모델을 훈련하세요:")
            print("  python test2.py --mode transfer_qat")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 렌더링 모드로 환경 재생성
        env.close()
        env = gym.make('CartPole-v1', render_mode='human')
        
        # 실시간 시뮬레이션 실행
        normal_data, qat_data = compare_live_simulations(normal_agent, qat_agent, env, episodes=3)
        
        print("\n실시간 시뮬레이션이 완료되었습니다!")
        print(f"결과 파일: {RESULTS_DIR}/live_simulation_comparison.png")
        
    elif args.mode == 'animate':
        print("\n=== 애니메이션 비교 모드 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다.")
            print("먼저 모델을 훈련하세요:")
            print("  python test2.py --mode transfer_qat")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 시뮬레이션 데이터 생성 (렌더링 없이)
        print("시뮬레이션 데이터 생성 중...")
        normal_data = run_live_simulation(normal_agent, env, "Normal DQN", episodes=1, delay=0)
        qat_data = run_live_simulation(qat_agent, env, "QAT DQN", episodes=1, delay=0)
        
        # 애니메이션 생성
        print("애니메이션 생성 중...")
        anim = create_animated_comparison(normal_data, qat_data)
        
        print("애니메이션 비교가 완료되었습니다!")
        print(f"결과 파일: {RESULTS_DIR}/cartpole_comparison.gif")
    
    env.close()
    print("\n프로그램이 완료되었습니다.")
    print(f"모든 결과 파일은 '{RESULTS_DIR}/' 폴더에서 확인할 수 있습니다.")
    
    # 다음 단계 안내
    if args.mode == 'transfer_qat':
        print("\n🎉 QAT 전이 학습 완료!")
        print("다음 단계:")
        print("  python test2.py --mode compare     # 상세 분석")
        print("  python test2.py --mode simulate    # 실시간 비교")
    elif args.mode == 'train_normal':
        print("\n✅ Normal DQN 훈련 완료!")
        print("다음 단계:")
        print("  python test2.py --mode transfer_qat --episodes 1000")
    elif args.mode == 'compare':
        print("\n📊 분석 완료! results/ 폴더의 차트들을 확인해보세요.")

# 추가 유틸리티 함수들
def compare_model_sizes():
    """모델 크기 비교 유틸리티"""
    if os.path.exists(NORMAL_MODEL_PATH) and os.path.exists(QAT_MODEL_PATH):
        normal_size = os.path.getsize(NORMAL_MODEL_PATH) / (1024 * 1024)
        qat_size = os.path.getsize(QAT_MODEL_PATH) / (1024 * 1024)
        
        print(f"\n📊 모델 파일 크기 비교:")
        print(f"Normal DQN: {normal_size:.3f} MB")
        print(f"QAT DQN:    {qat_size:.3f} MB")
        print(f"압축률:     {normal_size/qat_size:.2f}x")
        
        return normal_size, qat_size
    else:
        print("모델 파일을 찾을 수 없습니다.")
        return None, None

def quick_performance_test():
    """빠른 성능 테스트"""
    if not (os.path.exists(NORMAL_MODEL_PATH) and os.path.exists(QAT_MODEL_PATH)):
        print("모델 파일을 찾을 수 없습니다.")
        return
    
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # 모델 로드
    normal_agent = DQNAgent(state_size, action_size, use_qat=False)
    normal_agent.load(NORMAL_MODEL_PATH)
    
    qat_agent = DQNAgent(state_size, action_size, use_qat=True)
    qat_agent.load(QAT_MODEL_PATH)
    
    # 빠른 테스트 (20 에피소드)
    print("빠른 성능 테스트 중...")
    normal_scores = evaluate_agent(normal_agent, env, episodes=20)
    qat_scores = evaluate_agent(qat_agent, env, episodes=20)
    
    print(f"\n📊 빠른 성능 비교 (20 에피소드):")
    print(f"Normal DQN: {np.mean(normal_scores):.1f} ± {np.std(normal_scores):.1f}")
    print(f"QAT DQN:    {np.mean(qat_scores):.1f} ± {np.std(qat_scores):.1f}")
    print(f"성능 유지율: {(np.mean(qat_scores)/np.mean(normal_scores)*100):.1f}%")
    
    env.close()
    
    if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
        quick_performance_test()
    elif __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--size-check':
        compare_model_sizes()
        print("\n=== 두 모델 모두 훈련 ===")
        
        # Normal DQN 훈련
        print("\n1. 일반 DQN 훈련 시작...")
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        train_agent(normal_agent, env, episodes=TRAIN_EPISODES, agent_name="Normal DQN")
        normal_agent.save(NORMAL_MODEL_PATH)
        
        # QAT DQN 훈련
        print("\n2. QAT DQN 훈련 시작...")
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        train_agent(qat_agent, env, episodes=TRAIN_EPISODES, agent_name="QAT DQN")
        qat_agent.save(QAT_MODEL_PATH)
        
        print("\n두 모델 훈련이 완료되었습니다!")
        
    elif args.mode == 'compare':
        print("\n=== 모델 비교 분석 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다. 먼저 train_both 모드로 훈련을 실행하세요.")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 비교 분석 실행
        comparator = ModelComparator(normal_agent, qat_agent)
        comparator.generate_comprehensive_report(env)
        
    elif args.mode == 'simulate':
        print("\n=== 실시간 시뮬레이션 모드 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다. 먼저 train_both 모드로 훈련을 실행하세요.")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 렌더링 모드로 환경 재생성
        env.close()
        env = gym.make('CartPole-v1', render_mode='human')
        
        # 실시간 시뮬레이션 실행
        normal_data, qat_data = compare_live_simulations(normal_agent, qat_agent, env, episodes=3)
        
        print("\n실시간 시뮬레이션이 완료되었습니다!")
        print(f"결과 파일: {RESULTS_DIR}/live_simulation_comparison.png")
        
    elif args.mode == 'animate':
        print("\n=== 애니메이션 비교 모드 ===")
        
        # 모델 로드
        if not os.path.exists(NORMAL_MODEL_PATH) or not os.path.exists(QAT_MODEL_PATH):
            print("훈련된 모델을 찾을 수 없습니다. 먼저 train_both 모드로 훈련을 실행하세요.")
            sys.exit(1)
        
        normal_agent = DQNAgent(state_size, action_size, use_qat=False)
        normal_agent.load(NORMAL_MODEL_PATH)
        
        qat_agent = DQNAgent(state_size, action_size, use_qat=True)
        qat_agent.load(QAT_MODEL_PATH)
        
        # 시뮬레이션 데이터 생성 (렌더링 없이)
        print("시뮬레이션 데이터 생성 중...")
        normal_data = run_live_simulation(normal_agent, env, "Normal DQN", episodes=1, delay=0)
        qat_data = run_live_simulation(qat_agent, env, "QAT DQN", episodes=1, delay=0)
        
        # 애니메이션 생성
        print("애니메이션 생성 중...")
        anim = create_animated_comparison(normal_data, qat_data)
        
        print("애니메이션 비교가 완료되었습니다!")
        print(f"결과 파일: {RESULTS_DIR}/cartpole_comparison.gif")
    
    env.close()
    print("\n프로그램이 완료되었습니다.")
    print(f"모든 결과 파일은 '{RESULTS_DIR}/' 폴더에서 확인할 수 있습니다.")
