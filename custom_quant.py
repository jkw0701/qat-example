# Complete Custom Quantized Agent Pipeline with Simulation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import os
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import seaborn as sns
import time
from collections import deque

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 기존 Custom Quantized Agent 코드들 (custom_quant.py에서 가져옴)
class UniformAffineQuantizer(nn.Module):
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, 
                 scale_method: str = 'max', leaf_param: bool = False, weight_tensor=None, need_init=True):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        if weight_tensor is not None:
            self.inited = True
            if len(weight_tensor.shape) == 4:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1)))
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1, 1, 1)))
            elif len(weight_tensor.shape) == 2:
                self.delta = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))
                self.zero_point = nn.Parameter(torch.randn(size=(weight_tensor.shape[0], 1)))           
            else:
                print(weight_tensor.shape)
                raise ValueError('shape not implemented')
        else:
            self.inited = not need_init
            self.delta = nn.Parameter(torch.tensor(0.005))
            self.zero_point = nn.Parameter(torch.tensor(0.005))
    
    def init_quantization_scale(self, x, channel_wise=False):
        if channel_wise:
            if len(x.shape) == 4:
                delta = x.abs().max(dim=(1,2,3), keepdim=True)[0] / (self.n_levels - 1)
                zero_point = torch.zeros_like(delta)
            elif len(x.shape) == 2:
                delta = x.abs().max(dim=1, keepdim=True)[0] / (self.n_levels - 1)
                zero_point = torch.zeros_like(delta)
            else:
                delta = x.abs().max() / (self.n_levels - 1)
                zero_point = torch.tensor(0.0)
        else:
            if self.sym:
                delta = x.abs().max() / (self.n_levels / 2 - 1)
                zero_point = torch.tensor(self.n_levels / 2)
            else:
                x_min, x_max = x.min(), x.max()
                delta = (x_max - x_min) / (self.n_levels - 1)
                zero_point = -x_min / delta
                zero_point = torch.clamp(zero_point, 0, self.n_levels - 1)
        
        return delta, zero_point
    
    def clipping(self, x, lower, upper):
        x = x + F.relu(lower - x)
        x = x - F.relu(x - upper)
        return x
    
    def forward(self, x: torch.Tensor):
        if self.inited is False:
            delta, zero_point = self.init_quantization_scale(x, self.channel_wise)
            if not isinstance(zero_point, torch.Tensor):
                zero_point = torch.tensor(float(zero_point))
            self.delta = torch.nn.Parameter(delta)
            self.zero_point = torch.nn.Parameter(zero_point)
            self.inited = True

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = self.clipping(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

def round_ste(x):
    return (torch.round(x) - x).detach() + x

class CustomQuantizedDQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128, n_bits=8, symmetric=False):
        super(CustomQuantizedDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.n_bits = n_bits
        self.symmetric = symmetric
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
        
        self.weight_quantizer_fc1 = UniformAffineQuantizer(
            n_bits=n_bits, symmetric=symmetric, weight_tensor=self.fc1.weight, need_init=False
        )
        self.weight_quantizer_fc2 = UniformAffineQuantizer(
            n_bits=n_bits, symmetric=symmetric, weight_tensor=self.fc2.weight, need_init=False
        )
        self.weight_quantizer_fc3 = UniformAffineQuantizer(
            n_bits=n_bits, symmetric=symmetric, weight_tensor=self.fc3.weight, need_init=False
        )
        
        self.act_quantizer_input = UniformAffineQuantizer(n_bits=n_bits, symmetric=symmetric)
        self.act_quantizer1 = UniformAffineQuantizer(n_bits=n_bits, symmetric=symmetric)
        self.act_quantizer2 = UniformAffineQuantizer(n_bits=n_bits, symmetric=symmetric)
        
        print(f"✅ Custom Quantized DQN initialized: {n_bits}-bit, symmetric={symmetric}")
    
    def forward(self, x):
        x = self.act_quantizer_input(x)
        
        weight1_q = self.weight_quantizer_fc1(self.fc1.weight)
        x = F.linear(x, weight1_q, self.fc1.bias)
        x = self.relu(x)
        x = self.act_quantizer1(x)
        
        weight2_q = self.weight_quantizer_fc2(self.fc2.weight)
        x = F.linear(x, weight2_q, self.fc2.bias)
        x = self.relu(x)
        x = self.act_quantizer2(x)
        
        weight3_q = self.weight_quantizer_fc3(self.fc3.weight)
        x = F.linear(x, weight3_q, self.fc3.bias)
        
        return x
    
    def get_quantization_info(self):
        info = {
            'n_bits': self.n_bits,
            'symmetric': self.symmetric,
            'weight_quantizers': {},
            'activation_quantizers': {}
        }
        
        for name, quantizer in [
            ('fc1', self.weight_quantizer_fc1),
            ('fc2', self.weight_quantizer_fc2), 
            ('fc3', self.weight_quantizer_fc3)
        ]:
            if quantizer.delta is not None:
                info['weight_quantizers'][name] = {
                    'delta': quantizer.delta.item() if quantizer.delta.numel() == 1 else quantizer.delta.mean().item(),
                    'zero_point': quantizer.zero_point.item() if quantizer.zero_point.numel() == 1 else quantizer.zero_point.mean().item(),
                    'inited': quantizer.inited
                }
        
        for name, quantizer in [
            ('input', self.act_quantizer_input),
            ('act1', self.act_quantizer1),
            ('act2', self.act_quantizer2)
        ]:
            if quantizer.delta is not None:
                info['activation_quantizers'][name] = {
                    'delta': quantizer.delta.item() if quantizer.delta.numel() == 1 else quantizer.delta.mean().item(),
                    'zero_point': quantizer.zero_point.item() if quantizer.zero_point.numel() == 1 else quantizer.zero_point.mean().item(),
                    'inited': quantizer.inited
                }
        
        return info

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

class CustomQuantizedAgent:
    def __init__(self, state_size, action_size, lr=1e-3, n_bits=8, symmetric=False):
        self.state_size = state_size
        self.action_size = action_size
        self.n_bits = n_bits
        self.symmetric = symmetric
        
        print(f"🚀 CustomQuantizedAgent 초기화")
        print(f"   State size: {state_size}, Action size: {action_size}")
        print(f"   Learning rate: {lr}, Quantization: {n_bits}-bit")
        
        self.q_network = CustomQuantizedDQN(state_size, action_size, n_bits=n_bits, symmetric=symmetric)
        self.target_network = CustomQuantizedDQN(state_size, action_size, n_bits=n_bits, symmetric=symmetric)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.update_target_freq = 50
        self.step_count = 0
        
        self.memory = ReplayBuffer(10000)
        
        self.training_scores = []
        self.training_losses = []
        self.epsilon_history = []
        self.quantization_history = []
        
        print(f"✅ CustomQuantizedAgent 초기화 완료!")
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
            return np.argmax(q_values.cpu().numpy())
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def analyze_quantization(self):
        info = self.q_network.get_quantization_info()
        
        print(f"\n=== Quantization Analysis ===")
        print(f"Configuration: {info['n_bits']}-bit, symmetric={info['symmetric']}")
        
        print(f"\n📊 Weight Quantizers:")
        for layer_name, quant_info in info['weight_quantizers'].items():
            print(f"  {layer_name}: delta={quant_info['delta']:.6f}, zero_point={quant_info['zero_point']:.6f}")
        
        print(f"\n📈 Activation Quantizers:")
        for layer_name, quant_info in info['activation_quantizers'].items():
            print(f"  {layer_name}: delta={quant_info['delta']:.6f}, zero_point={quant_info['zero_point']:.6f}")
        
        self.quantization_history.append(info)
        return info
    
    def load_pretrained_weights(self, normal_model_path):
        print(f"\n🔄 가중치 전이 시작: {normal_model_path}")
        
        try:
            checkpoint = torch.load(normal_model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                normal_weights = checkpoint['model_state_dict']
                print(f"   새로운 형식 모델 파일 로드")
            else:
                normal_weights = checkpoint
                print(f"   기존 형식 모델 파일 로드")
            
            model_dict = self.q_network.state_dict()
            transferred_count = 0
            
            weight_mapping = {
                'fc1.weight': 'fc1.weight',
                'fc1.bias': 'fc1.bias',
                'fc2.weight': 'fc2.weight', 
                'fc2.bias': 'fc2.bias',
                'fc3.weight': 'fc3.weight',
                'fc3.bias': 'fc3.bias'
            }
            
            for quantized_key, normal_key in weight_mapping.items():
                if normal_key in normal_weights and quantized_key in model_dict:
                    model_dict[quantized_key] = normal_weights[normal_key].clone()
                    transferred_count += 1
                    print(f"   ✅ {normal_key} -> {quantized_key}")
                else:
                    print(f"   ❌ {normal_key} not found")
            
            self.q_network.load_state_dict(model_dict, strict=False)
            self.target_network.load_state_dict(model_dict, strict=False)
            
            print(f"✅ 가중치 전이 완료! ({transferred_count}/{len(weight_mapping)} 성공)")
            
        except Exception as e:
            print(f"❌ 가중치 전이 실패: {e}")
            print(f"   처음부터 훈련을 시작합니다.")
    
    def benchmark_inference_speed(self, num_samples=1000):
        self.q_network.eval()
        test_inputs = torch.randn(num_samples, self.state_size)
        
        with torch.no_grad():
            for _ in range(10):
                _ = self.q_network(test_inputs[:10])
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(0, num_samples, 32):
                batch = test_inputs[i:i+32]
                _ = self.q_network(batch)
        
        end_time = time.time()
        total_time = end_time - start_time
        throughput = num_samples / total_time
        
        return {
            'total_time': total_time,
            'avg_time_ms': (total_time / num_samples) * 1000,
            'throughput': throughput,
            'is_quantized': True,
            'device': 'cpu'
        }
    
    def get_model_size_mb(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.q_network.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.q_network.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            'model_state_dict': self.q_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'training_scores': self.training_scores,
            'training_losses': self.training_losses,
            'epsilon_history': self.epsilon_history,
            'quantization_history': self.quantization_history,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'n_bits': self.n_bits,
                'symmetric': self.symmetric,
                'epsilon': self.epsilon,
                'step_count': self.step_count
            }
        }
        
        torch.save(save_dict, path)
        print(f"✅ Custom quantized model saved to {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        if 'target_state_dict' in checkpoint:
            self.target_network.load_state_dict(checkpoint['target_state_dict'])
        
        self.training_scores = checkpoint.get('training_scores', [])
        self.training_losses = checkpoint.get('training_losses', [])
        self.epsilon_history = checkpoint.get('epsilon_history', [])
        self.quantization_history = checkpoint.get('quantization_history', [])
        
        if 'hyperparameters' in checkpoint:
            hyper = checkpoint['hyperparameters']
            self.epsilon = hyper.get('epsilon', self.epsilon)
            self.step_count = hyper.get('step_count', 0)
        
        self.q_network.eval()
        print(f"✅ Custom quantized model loaded from {path}")

# 🎯 훈련 함수
def train_custom_agent(agent, env, episodes=2000, agent_name="Custom Quantized DQN"):
    print(f"\n🚀 {agent_name} 훈련 시작 ({episodes} 에피소드)")
    
    scores = []
    scores_window = deque(maxlen=100)
    
    for episode in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
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
            
            if len(agent.memory) >= agent.batch_size:
                loss = agent.replay()
                if loss is not None:
                    episode_losses.append(loss)
        
        scores_window.append(total_reward)
        scores.append(total_reward)
        agent.training_scores.append(total_reward)
        agent.epsilon_history.append(agent.epsilon)
        
        if episode_losses:
            agent.training_losses.append(np.mean(episode_losses))
        
        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores_window)
            print(f"Episode {episode+1:3d}: Score={total_reward:3.0f}, Avg100={avg_score:5.1f}, ε={agent.epsilon:.3f}")
        
        if (episode + 1) % 200 == 0:
            agent.analyze_quantization()
    
    return scores

# 🎮 평가 함수  
def evaluate_agent(agent, env, episodes=100):
    scores = []
    original_epsilon = getattr(agent, 'epsilon', 0)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0
    
    for _ in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            state = next_state
            total_reward += reward
        
        scores.append(total_reward)
    
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon
    
    return scores

# 🎬 시뮬레이션 함수들
def run_simulation(agent, env, agent_name="Agent", episodes=3, delay=0.05, render=False):
    print(f"\n=== {agent_name} 시뮬레이션 ===")
    
    simulation_data = {
        'positions': [],
        'angles': [],
        'actions': [],
        'rewards': [],
        'episode_scores': []
    }
    
    original_epsilon = getattr(agent, 'epsilon', 0)
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0
    
    for episode in range(episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        total_reward = 0
        
        episode_positions = []
        episode_angles = []
        episode_actions = []
        episode_rewards = []
        
        print(f"에피소드 {episode + 1} 시작...")
        
        while True:
            if render and hasattr(env, 'render') and delay > 0:
                env.render()
                time.sleep(delay)
            
            cart_pos = state[0]
            pole_angle = state[2]
            episode_positions.append(cart_pos)
            episode_angles.append(pole_angle)
            
            action = agent.act(state)
            episode_actions.append(action)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            
            episode_rewards.append(reward)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        simulation_data['positions'].append(episode_positions)
        simulation_data['angles'].append(episode_angles)
        simulation_data['actions'].append(episode_actions)
        simulation_data['rewards'].append(episode_rewards)
        simulation_data['episode_scores'].append(total_reward)
        
        print(f"에피소드 {episode + 1} 완료 - 점수: {total_reward}, 스텝: {len(episode_positions)}")
    
    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon
    
    avg_score = np.mean(simulation_data['episode_scores'])
    print(f"{agent_name} 평균 점수: {avg_score:.2f}")
    
    return simulation_data

# 📊 시각화 함수들
def plot_training_results(agent):
    """훈련 결과 시각화"""
    os.makedirs("results", exist_ok=True)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Custom Quantized DQN Training Results', fontsize=16, fontweight='bold')
    
    # 1. 훈련 점수
    if agent.training_scores:
        episodes = range(len(agent.training_scores))
        ax1.plot(episodes, agent.training_scores, alpha=0.7, color='red', label='Custom Quantized DQN')
        
        if len(agent.training_scores) >= 50:
            window_size = 50
            moving_avg = np.convolve(agent.training_scores, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(agent.training_scores)), moving_avg, 
                    color='darkred', linewidth=2, label='Moving Average (50)')
        
        ax1.set_title('Training Scores')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 손실 함수
    if agent.training_losses:
        ax2.plot(agent.training_losses, color='red', alpha=0.7)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Loss')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon 감소
    if agent.epsilon_history:
        ax3.plot(agent.epsilon_history, color='green')
        ax3.set_title('Epsilon Decay')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.grid(True, alpha=0.3)
    
    # 4. Quantization 정보 (최신)
    if agent.quantization_history:
        latest_quant = agent.quantization_history[-1]
        
        layers = list(latest_quant['weight_quantizers'].keys())
        deltas = [latest_quant['weight_quantizers'][layer]['delta'] for layer in layers]
        zero_points = [latest_quant['weight_quantizers'][layer]['zero_point'] for layer in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        ax4.bar(x - width/2, deltas, width, label='Delta (Scale)', alpha=0.7)
        ax4.bar(x + width/2, zero_points, width, label='Zero Point', alpha=0.7)
        
        ax4.set_title('Quantization Parameters (Latest)')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(layers)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/custom_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_simulation_comparison(normal_data, custom_data):
    """Normal vs Custom 시뮬레이션 비교"""
    os.makedirs("results", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Normal vs Custom Quantized DQN Simulation Comparison', fontsize=16, fontweight='bold')
    
    # 1. 에피소드 점수 비교
    normal_scores = normal_data['episode_scores']
    custom_scores = custom_data['episode_scores']
    
    episodes = range(1, len(normal_scores) + 1)
    width = 0.35
    x = np.arange(len(episodes))
    
    axes[0,0].bar(x - width/2, normal_scores, width, label='Normal DQN', color='blue', alpha=0.7)
    axes[0,0].bar(x + width/2, custom_scores, width, label='Custom Quantized DQN', color='red', alpha=0.7)
    axes[0,0].set_title('Episode Scores Comparison')
    axes[0,0].set_xlabel('Episode')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([f'Ep{i}' for i in episodes])
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 카트 위치 궤적 (첫 번째 에피소드)
    if normal_data['positions'] and custom_data['positions']:
        normal_pos = normal_data['positions'][0]
        custom_pos = custom_data['positions'][0]
        
        axes[0,1].plot(range(len(normal_pos)), normal_pos, label='Normal DQN', color='blue', alpha=0.7)
        axes[0,1].plot(range(len(custom_pos)), custom_pos, label='Custom Quantized DQN', color='red', alpha=0.7)
        axes[0,1].axhline(y=2.4, color='orange', linestyle='--', alpha=0.5, label='Boundary')
        axes[0,1].axhline(y=-2.4, color='orange', linestyle='--', alpha=0.5)
        axes[0,1].set_title('Cart Position Trajectory (Episode 1)')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Cart Position')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. 폴 각도 궤적 (첫 번째 에피소드)
    if normal_data['angles'] and custom_data['angles']:
        normal_angles = np.degrees(normal_data['angles'][0])
        custom_angles = np.degrees(custom_data['angles'][0])
        
        axes[1,0].plot(range(len(normal_angles)), normal_angles, label='Normal DQN', color='blue', alpha=0.7)
        axes[1,0].plot(range(len(custom_angles)), custom_angles, label='Custom Quantized DQN', color='red', alpha=0.7)
        axes[1,0].axhline(y=12, color='orange', linestyle='--', alpha=0.5, label='Boundary')
        axes[1,0].axhline(y=-12, color='orange', linestyle='--', alpha=0.5)
        axes[1,0].set_title('Pole Angle Trajectory (Episode 1)')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Pole Angle (degrees)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. 성능 통계 비교
    normal_avg = np.mean(normal_scores)
    normal_std = np.std(normal_scores)
    custom_avg = np.mean(custom_scores)
    custom_std = np.std(custom_scores)
    
    models = ['Normal DQN', 'Custom Quantized DQN']
    averages = [normal_avg, custom_avg]
    stds = [normal_std, custom_std]
    colors = ['blue', 'red']
    
    bars = axes[1,1].bar(models, averages, color=colors, alpha=0.7, yerr=stds, capsize=5)
    axes[1,1].set_title('Performance Comparison')
    axes[1,1].set_ylabel('Average Score ± Std')
    axes[1,1].grid(True, alpha=0.3)
    
    for bar, avg, std in zip(bars, averages, stds):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 5,
                      f'{avg:.1f}±{std:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/simulation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_animation(normal_data, custom_data):
    """Normal vs Custom 애니메이션 생성"""
    os.makedirs("results", exist_ok=True)
    
    if not normal_data['positions'] or not custom_data['positions']:
        print("애니메이션을 위한 데이터가 부족합니다.")
        return
    
    normal_pos = normal_data['positions'][0]
    normal_angles = normal_data['angles'][0]
    custom_pos = custom_data['positions'][0]
    custom_angles = custom_data['angles'][0]
    
    max_steps = max(len(normal_pos), len(custom_pos))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Real-time CartPole Comparison Animation', fontsize=16, fontweight='bold')
    
    # Normal DQN 서브플롯
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-0.5, 2)
    ax1.set_title('Normal DQN')
    ax1.set_xlabel('Position')
    ax1.grid(True, alpha=0.3)
    
    # Custom DQN 서브플롯
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-0.5, 2)
    ax2.set_title('Custom Quantized DQN')
    ax2.set_xlabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # 그래픽 요소 초기화
    cart1 = Rectangle((-0.25, 0), 0.5, 0.3, fc='blue', alpha=0.7)
    pole1, = ax1.plot([], [], 'b-', linewidth=8)
    ax1.add_patch(cart1)
    
    cart2 = Rectangle((-0.25, 0), 0.5, 0.3, fc='red', alpha=0.7)
    pole2, = ax2.plot([], [], 'r-', linewidth=8)
    ax2.add_patch(cart2)
    
    score_text1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    score_text2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        # Normal DQN 업데이트
        if frame < len(normal_pos):
            cart_x1 = normal_pos[frame]
            pole_angle1 = normal_angles[frame]
            
            cart1.set_x(cart_x1 - 0.25)
            pole_x1 = [cart_x1, cart_x1 + np.sin(pole_angle1)]
            pole_y1 = [0.15, 0.15 + np.cos(pole_angle1)]
            pole1.set_data(pole_x1, pole_y1)
            score_text1.set_text(f'Step: {frame + 1}\nScore: {frame + 1}')
        
        # Custom DQN 업데이트
        if frame < len(custom_pos):
            cart_x2 = custom_pos[frame]
            pole_angle2 = custom_angles[frame]
            
            cart2.set_x(cart_x2 - 0.25)
            pole_x2 = [cart_x2, cart_x2 + np.sin(pole_angle2)]
            pole_y2 = [0.15, 0.15 + np.cos(pole_angle2)]
            pole2.set_data(pole_x2, pole_y2)
            score_text2.set_text(f'Step: {frame + 1}\nScore: {frame + 1}')
        
        return cart1, pole1, cart2, pole2, score_text1, score_text2
    
    anim = animation.FuncAnimation(fig, animate, frames=max_steps, 
                                 interval=100, blit=True, repeat=True)
    
    plt.tight_layout()
    
    try:
        anim.save('results/cartpole_comparison.gif', writer='pillow', fps=10)
        print("🎬 애니메이션이 'results/cartpole_comparison.gif'에 저장되었습니다.")
    except:
        print("GIF 저장에 실패했습니다. pillow 패키지가 필요합니다.")
    
    plt.show()
    return anim

# 🎯 메인 파이프라인 함수
def run_complete_pipeline():
    """완전한 파이프라인 실행: 가중치 전이 → 훈련 → 평가 → 시뮬레이션"""
    
    print("🚀 Complete Custom Quantized DQN Pipeline")
    print("="*60)
    
    # 환경 설정
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # 1. Normal 모델 파일 확인
    NORMAL_MODEL_PATH = "models/dqn_normal.pth"
    if not os.path.exists(NORMAL_MODEL_PATH):
        print(f"❌ Normal 모델을 찾을 수 없습니다: {NORMAL_MODEL_PATH}")
        print("먼저 Normal DQN을 훈련하세요:")
        print("python test2.py --mode train_normal")
        return
    
    print(f"✅ Normal 모델 발견: {NORMAL_MODEL_PATH}")
    
    # 2. Custom Quantized Agent 생성
    print(f"\n📦 Custom Quantized Agent 생성 중...")
    custom_agent = CustomQuantizedAgent(
        state_size=state_size,
        action_size=action_size,
        lr=1e-3,
        n_bits=8,
        symmetric=False
    )
    
    # 3. 가중치 전이
    print(f"\n🔄 가중치 전이...")
    custom_agent.load_pretrained_weights(NORMAL_MODEL_PATH)
    
    # 4. 전이 후 초기 성능 평가
    print(f"\n📊 전이 후 초기 성능 평가...")
    initial_scores = evaluate_agent(custom_agent, env, episodes=50)
    initial_avg = np.mean(initial_scores)
    print(f"전이 후 초기 성능: {initial_avg:.2f} ± {np.std(initial_scores):.2f}")
    
    # 5. 훈련 실행
    print(f"\n🎯 Custom Quantized DQN 훈련...")
    training_scores = train_custom_agent(custom_agent, env, episodes=500)
    
    # 6. 훈련 후 성능 평가
    print(f"\n📈 훈련 후 성능 평가...")
    final_scores = evaluate_agent(custom_agent, env, episodes=100)
    final_avg = np.mean(final_scores)
    print(f"훈련 후 성능: {final_avg:.2f} ± {np.std(final_scores):.2f}")
    
    # 7. 모델 저장
    print(f"\n💾 모델 저장...")
    os.makedirs("models", exist_ok=True)
    CUSTOM_MODEL_PATH = "models/dqn_custom.pth"
    custom_agent.save(CUSTOM_MODEL_PATH)
    
    # 8. 훈련 결과 시각화
    print(f"\n📊 훈련 결과 시각화...")
    plot_training_results(custom_agent)
    
    # 9. Normal 모델과 비교를 위한 로드 (간단한 더미 구현)
    print(f"\n🔄 Normal 모델과 성능 비교를 위한 시뮬레이션...")
    
    # Normal 모델 더미 데이터 (실제로는 Normal Agent를 로드해야 함)
    # 여기서는 간단히 더미 데이터로 시연
    dummy_normal_data = {
        'episode_scores': [500, 500, 500],  # 더미 데이터
        'positions': [np.random.randn(200) * 0.5],
        'angles': [np.random.randn(200) * 0.1],
        'actions': [np.random.randint(0, 2, 200)],
        'rewards': [np.ones(200)]
    }
    
    # 10. Custom 모델 시뮬레이션
    print(f"\n🎮 Custom 모델 시뮬레이션...")
    custom_sim_data = run_simulation(custom_agent, env, "Custom Quantized DQN", episodes=3)
    
    # 11. 비교 시각화
    print(f"\n📊 비교 시각화...")
    plot_simulation_comparison(dummy_normal_data, custom_sim_data)
    
    # 12. 애니메이션 생성
    print(f"\n🎬 애니메이션 생성...")
    create_animation(dummy_normal_data, custom_sim_data)
    
    # 13. 최종 결과 요약
    print(f"\n" + "="*60)
    print(f"🎉 Complete Pipeline 완료!")
    print(f"="*60)
    print(f"📈 성능 개선: {initial_avg:.2f} → {final_avg:.2f}")
    print(f"📦 모델 저장: {CUSTOM_MODEL_PATH}")
    print(f"📊 결과 파일:")
    print(f"   - results/custom_training_results.png")
    print(f"   - results/simulation_comparison.png") 
    print(f"   - results/cartpole_comparison.gif")
    print(f"🔧 모델 정보:")
    print(f"   - Quantization: {custom_agent.n_bits}-bit")
    print(f"   - 모델 크기: {custom_agent.get_model_size_mb():.3f} MB")
    
    # 14. Quantization 최종 분석
    print(f"\n🔬 최종 Quantization 분석:")
    custom_agent.analyze_quantization()
    
    env.close()
    return custom_agent

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Custom Quantized DQN Complete Pipeline')
    parser.add_argument('--mode', choices=['pipeline', 'test'], default='pipeline',
                       help='실행 모드: pipeline(전체 실행) 또는 test(테스트)')
    
    args = parser.parse_args()
    
    if args.mode == 'pipeline':
        run_complete_pipeline()
    else:
        # 테스트 모드
        print("🧪 테스트 모드")
        agent = CustomQuantizedAgent(4, 2, n_bits=8)
        
        # 더미 데이터로 테스트
        for _ in range(100):
            state = np.random.randn(4)
            action = np.random.randint(2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = np.random.rand() > 0.9
            agent.remember(state, action, reward, next_state, done)
        
        # 학습 테스트
        for _ in range(10):
            loss = agent.replay()
            if loss:
                print(f"Loss: {loss:.6f}")
        
        print("✅ 테스트 완료!")