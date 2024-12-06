# Training_Pusher_by_DQN

## 목차
- [Training\_Pusher\_by\_DQN](#training_pusher_by_dqn)
  - [목차](#목차)
  - [실행 방법](#실행-방법)
  - [프로젝트 동기](#프로젝트-동기)
  - [프로젝트 개요](#프로젝트-개요)
  - [pusher 환경 설명](#pusher-환경-설명)
    - [action](#action)
    - [observation](#observation)
    - [reward](#reward)
    - [starting state](#starting-state)
    - [episode end](#episode-end)
  - [Q-learning으로 시도 및 깨달음](#q-learning으로-시도-및-깨달음)
  - [코드 설명](#코드-설명)
  - [코드 실행 결과 및 물리엔진 mujoco의 버그](#코드-실행-결과-및-물리엔진-mujoco의-버그)
  - [버그 해결 후 재학습 결과](#버그-해결-후-재학습-결과)
    - [버그 있는 채로 학습했던 모델에 이어서 학습시킨 결과](#버그-있는-채로-학습했던-모델에-이어서-학습시킨-결과)
    - [새롭게 학습시킨 결과](#새롭게-학습시킨-결과)
  - [결과 부족한 부분 해결책](#결과-부족한-부분-해결책)
  - [기타 흥미로운 내용](#기타-흥미로운-내용)
  - [느낀 점](#느낀-점)
  - [참고자료](#참고자료)

## [실행 방법](#목차)
colab에서 진행하면 됩니다. repo에 있는 Training_Pusher_by_DQN 폴더를 그대로 구글 드라이브의 ‘내 드라이브’에 업로드하면 됩니다. (코드 변경 없이 내부 디렉토리를 수정하면 실행시 에러가 발생합니다) 
코드도 DQN.ipynb로 폴더에 함께 포함돼있습니다. 설치해야할 패키지들은 모두 코드로 포함돼있고 colab 상에서 설치됩니다.

## [프로젝트 동기](#목차)
로봇을 다루는 기계시스템디자인공학과에 입학할 때부터, 로봇을 제어하기 위해 인간이 동작 하나하나를 작성해줘야 한다는 점이 마음에 들지 않았다.  로봇이 인간처럼 스스로 학습해나갈 수 있으면 좋겠다는 생각을 했었다. 그래서 규칙에 따라 스스로 학습해 나가는 메커니즘인 강화학습을 처음 접하고 망설임 없이 공부해봤었다. 그리고 이번 기회에 시뮬레이션을 통해나마 로봇이 스스로 자연스러운 행동을 학습해나가는 걸 직접 구현해보고 싶어 해당 주제를 선정하게 됐다.

## [프로젝트 개요](#목차)
gymnasium의 pusher를 성공적으로 학습시키는 게 목표다.

![pusher](https://github.com/user-attachments/assets/e1ad2d0b-b980-470b-a9f6-59493220f4d1)

Pusher는 위 동영상과 같이 다관절 로봇으로, 흰색 물체를 빨간색 지점으로 옮기는 게 목표인 agent다.

openAI의 gymnasium에서 물리 엔진 mujoco를 이용해 시뮬레이션을 지원하기 때문에, 우리는 ‘mujoco측에서 매 step 시뮬레이션 결과로써 반환하는 observation, reward 등의 정보’를 이용해, 이번 step에 어떤 action을 취할지에 대한 결정만 env.step() 함수를 통해 gymnasium 환경에 넘겨주면 된다.
`observation, reward, terminated, truncated, info = env.step(action) `
 
 
처음에는 Q-learning을 이용해 학습시켜보려 했지만( [Q-learning으로 시도 및 깨달음](#q-learning으로-시도-및-깨달음) 부분 참고), 그게 불가능하다는 걸 깨닫고 DQN을 이용해 학습시켰다.

## [pusher 환경 설명](#목차)

### action
Action Space는 shape이 (7,), (최솟값,최댓값)은 (-2.0, 2.0), 데이터형은 float32이다.

![image](https://github.com/user-attachments/assets/5d7fdc1e-0cef-45d0-a423-52498ead72e9)

action의 종류는 위 캡처와 같다.
0의 경우, 어깨를 이용해 팔을 좌우로 움직이는 것, 1은 어깨를 이용해 팔을 올리고 내리는 것, 2는 어깨를 돌리는 것, 3은 팔꿈치를 기준으로 전완을 굽히고 펴는 것, 4는 팔꿈치를 기준으로 전완을 돌리는 것, 5는 손목을 굽히고 펴는 것, 6은 손목을 돌리는 걸 가리킨다.

### observation
![image](https://github.com/user-attachments/assets/7188525e-a9b9-4472-8edd-248212ed2b0c)

Observation Space는 shape이 (23,), (최솟값,최댓값)은 (-inf, inf), 데이터형은 float64다,
Observation의 0\~6은 현재 joint들이 얼마나 회전된 상태인지를 나타낸다. 여기서 0\~6이 맡는 joint는 action 0\~6이 의미하는 joint와 순서가 일치한다.
Observation의 7\~13은 joint들의 각속도를 나타내며, 마찬가지로 7\~13이 맡는 joint 순서는 action 0\~6의 joint 순서와 같다.
14\~16은 로봇의 손가락 위치의 x,y,z 좌표다.
17\~19는 옮겨야할 물건 위치의 x,y,z 좌표다.
20\~22는 물건이 옮겨져야할 목표 위치의 x,y,z 좌표다.

### reward
![image](https://github.com/user-attachments/assets/16c9e04d-3111-4c65-abe8-19d650aaedc4)

Rewards는 세 가지로 이뤄져 있다. 
1. reward_near : -(로봇의 fingertip과 목표지점의 거리)
2. reward_dist : -(물체와 목표지점의 거리)
3. reward_control : -(action의 크기) 로, 7가지 action을 각각 제곱해서 더함으로써 action의 크기를 구함.
그리고 전체 reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near 와 같이 가중치를 부여하여 각각을 합친다.
Reward의 모든 요소는 로봇이 얼마나 못하고 있는지를 나타내는 형태로, 항상 음수이며, 매 step마다 무조건 reward를 부여하는 환경임을 알 수 있다.
그리고 reward 요소들의 내용을 보면, 매 순간순간 agent가 수행을 얼마나 잘하고 있는지, 직전에 비해 더 나은 방향으로 가고 있는지를 잘 알려줌을 알 수 있다.

### starting state
![image](https://github.com/user-attachments/assets/a002b970-1e0a-4f81-bab0-3b72529207ea)

환경이 시작될 때, 로봇 관절들의 각변위는 항상 0이며, 각속도는 0을 기준으로 uniform noise [-0.005, 0.005]를 갖고 시작한다.
물체의 x좌표는 [-0.3, 0]에서, y좌표는 [-0.2, 0.2]에서 랜덤하게 정해진다. 이때 (x,y)의 원점과의 거리가 0.17이 넘지 않게 정해진다.
목표지점의 좌표는 (0.45, -0.05, -0.323)로 고정이다.
한 번의 env.step(action)은 5프레임 간 지속된다. 1프레임은 0.01초이므로 한 action은 0.05초 간 지속된다. 

### episode end
![image](https://github.com/user-attachments/assets/2f6330a5-2a8c-4757-9d0a-0f070240d3f5)

Truncation은 한 episode가 지지부진하게 이어져서 시간만 낭비되는 걸 막고자 일정한 시간이 되면 episode를 그냥 끝내버리는 거다. 이 environment의 경우, 100번째 env.step이 끝나고 truncated 된다. 직전 페이지에서 한 step은 0.05초 지속된다 했으므로, 이 environment는 truncated 될 경우, 한 episode가 5초 동안 진행된다는 걸 알 수 있다.
Termination은 state space value가 infinite해질 때 된다고 써있다.
그런데 이 환경에서 action을 처음부터 끝까지 최대 크기로 해서 실행해본 결과, truncated 되는 걸 확인했다. Truncated 기준이 5초일 때는 state가 infinite해질 일이 없는 것이다. 즉, 이번 실습 상에서는 terminated 될 일이 없다. 항상 truncated 된다.


## [Q-learning으로 시도 및 깨달음](#목차)
처음에는 Q-learning을 시도했다. 그러나 Q-learning으로는 불가능하다는 걸 깨닫고 DQN을 이용해서 성공적으로 학습시켰다.

Q-learning으로 코드를 다 짜고 실행을 했다. 실행을 했는데 number of visted (s,a)가 episode수에 따라 선형적으로 증가하고, 대부분의 state는 한번 방문한 것으로 떴다. 그리고 7만번 정도의 episode를 돌고 나니 ram부족으로 런타임이 종료됐다. 이때 깨달았다. 애초에 pusher env를 해결하기 위해 q-learning을 쓰면 안 됐다. 이 env는 State space, action space는 총 23+7=30 개의 인자들로 이뤄져 있다. 그리고 각 인자는 -inf\~inf 또는 -2\~2의 연속형 변수다. 그런데 각 인자에서 고작 10개의 discrete한 수준값만 시도한다고 해도 (s,a) 경우의 수는 10^30이 된다. 해당 경우들의 Q값만 저장해도 4\*10^30 byte = 4\*10^18 byte가 필요하다. 터무니 없는 수치다. Q-learning이 state, action이 복잡하면 사용할 수 없는 알고리즘이란 걸 당연히 알고는 있었지만, 그 말의 의미를 깊게 생각해보지 않은 결과였다. 앞으로는 알고리즘을 적용할 때 미리 공간복잡도를 꼭 따져봐야겠다는 다짐을 강하게 했다.

dqn의 위력을 실감하고, dqn을 이용해 새로운 마음으로 다시 시작했다.

## [코드 설명](#목차)
dqn의 output으로 둘 action들을 정해야 한다. 
이 env는 앞서 설명했듯 7개의 연속형 변수들로 action space가 구성된다. 
dqn에서 discrete하게 사용할 action을 적절하게 선정해야 한다.
우선, 한 step에 한 종류의 joint만 움직이는 걸로 정했다. 즉, 7개의 변수 중 1개의 변수만 0이 아니게 둔다. 
그리고 action 크기는, 가능한 범위가 -2~2이기 때문에, -1, 1만 둬도 충분하겠다고 판단했다. 
따라서 사용할 action은 (1,0,0,0,0,0,0),(-1,0,0,0,0,0,0), … , (0,0,0,0,0,0,1),(0,0,0,0,0,0,-1)의 14가지가 된다.

```python
# actions shape (batch,14) numpy array
actions = dqn(tf.convert_to_tensor([obs], dtype=tf.float32))
# action shape (1,) numpy array
action = np.argmax(actions.numpy(), axis=-1)

action = epsilon_greedy(action, global_step)

# action을 env에 맞춰 shape (7,)로 변환
action_env=change_action_format(action)

next_obs,reward,terminated,truncated,info = env.step(action_env)
```
mainQ에서 shape (1,14) 형태로 반환된 q값들을 np.argmax를 통해 하나의 action (1,)로 변환하고, 이를 epsilon-greedy에도 사용한 후, env.step에서는 action을 (7,)형태로 요구한다. 따라서 이를 실행하기 전 action의 형태를 변환할 필요가 있다. 
이를 위해 change_action_format()을 추가했다.
 
```python
def change_action_format(action):
  action=action[0]
  action_type=action//2
  action_sign=action%2
  list=np.zeros(shape=(7,),dtype=np.float32)
  if action_sign:
    list[action_type]=-1.0
  else:
    list[action_type]=1.0
  return list
```
action [0]을 [1,0,0,0,0,0,0]로, [1]을 [-1,0,0,0,0,0,0]로, [12]를 [0,0,0,0,0,0,1]로, [13]을 [0,0,0,0,0,0,-1]로 변환하는 식으로 작동한다.


이제 작성한 코드를 위에서부터 차례로 보고자 한다. (설명할 내용이 없는 부분은 제외)

```python
from google.colab import drive
drive.mount('/content/drive')

FOLDERNAME = 'Training_Pusher_by_DQN/DQN'

%cd /content/drive/MyDrive/$FOLDERNAME
```
코랩은 클라우드 환경이므로 런타임이 종료되면 훈련시킨 결과들이 날라간다. 결과들을 보존하려면 drive에 저장해야 한다.
위 코드가 실행되려면 드라이브에 Training_Pusher_by_DQN 폴더, 그 아래 DQN 폴더가 있어야 한다.
상대경로를 편하게 쓰고자 DQN 폴더까지로 change directory도 해둔다.

```python
! pip install gymnasium[mujoco]==0.29.1 tensorflow==2.15.0 dill
# pusher-v4에서 mujoco>=3.0.0을 쓰면 물체에 fingertip이 닿지 않고 뚫고 지나감
! pip install mujoco==2.3.7
```
필요한 패키지를 설치한다. 
mujoco는 처음에 그냥 gymnasium에 딸려있는 3.x 버전을 썼는데, 훈련 중에 물체에 로봇의 fingertip이 닿지 않고 뚫고 지나가는 에러가 포함돼있었다는 사실을 훈련 결과 비디오를 보고 파악하고, 해결하는 과정에서 추가했다. ([코드 실행 결과 및 물리엔진 mujoco의 버그](#목차))

```python
# 물건 닿지 않는 에러 해결 위해
DEFAULT_CAMERA_CONFIG = {
    'trackbodyid': -1,
    'distance': 3.0,
    'azimuth': 135.0,
    'elevation': -22.5,
}

env = gym.make("Pusher-v4", render_mode="rgb_array")
env.unwrapped.mujoco_renderer.default_cam_config = DEFAULT_CAMERA_CONFIG

n_outputs = 14
```
물체가 fingertip에 닿지 않는 에러를 해결하기 위해서는 로봇 행동을 담는 카메라를 비스듬하고 낮은 각도로 바꿔야 했다. (아래는 바꾸기 전후 모습)
n_outputs는 dqn에서 output의 갯수다. 앞서 설명했듯 14가지로 정했다.
   

```python
eps_min = 0.3
eps_max = 0.5
eps_decay_steps = 100000

def epsilon_greedy(action, step):
  epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
  if np.random.rand() < epsilon:
    return np.array([np.random.randint(0,n_outputs)])
  else:
    return action
```
Epsilon_greedy는 학습 초반에는 아무것도 모르는 agent가 본인이 최적이라 생각하는 것만 많이 시도하는 게 좋은 결과를 내지 못할 것 같아 위처럼 0.8\~1의 높은 epsilon 값을 이용했다.
그후 episodic loss가 그래프상 안정됐다고 판단될 때 0.3\~0.5으로 낮췄다.

```python
buffer_len = 20000
exp_buffer = deque(maxlen=buffer_len)
```
학습에 쓸 데이터들을 저장해두는 buffer다. 
```python
def sample_memories(batch_size):
    perm_batch = np.random.permutation(len(exp_buffer))[:batch_size]
    #포인터로 저장
    mem = np.array(exp_buffer,dtype=object)[perm_batch]
    return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]
```
이들 중 랜덤으로 batch_size만큼 골라 각 학습 step에 사용한다.

```python
# A larger batch size can reduce the variance in the gradient updates.
batch_size = 100
learning_rate = 0.001
X_shape = (None, 23)
discount_factor = 0.97

# 이 env는 항상 100step 돌고 끝남
one_episode_period=100
global_step = 0
copy_steps = 20
# one_episode_period(100)을 나눠떨어지게 하는 수로 해야 episode당 학습수 동일해서 episodic loss가 의미있어짐
steps_train = 4
# n번째 에피소드부터 학습, copy 시작
episode_to_start_train=5
start_steps = one_episode_period*(episode_to_start_train-1)

# train 결과 몇 개 episode 완료할 때마다 평균내서 출력할지
period_print_result=10
```
Batch_size가 클수록 gradient 업데이트가 보다 안정적으로 된다고 해서 100 정도로 잡았다.
X_shape은 dqn에 투입할 input의 크기로, None 자리는 1 또는 batch_size 가 오게 되고, 23은 observation space의 크기를 의미한다.
one_episode_period은 앞서 설명했듯 에피소드가 항상 100step을 밟고 truncation 되기 때문에 100이다.
global_step은 episode가 끝나더라도 step을 계속 누적해서, 학습 루프 시작 후 지금까지 겪은 모든 step 수를 뜻한다. 
copy_steps는 main network를 target network로 복사하는 주기다.
start_steps는 학습을 시작하는 global_step 으로, one_episode_period*(episode_to_start_train-1)으로 정함으로써, episode_to_start_train(번째) 에피소드의 첫 step부터 학습을 시작하도록 했다. 
그리고 학습을 하는 step 주기인 steps_train을 one_episode_period를 나눠떨어지게 하는 수로 정했다.
start_steps와 steps_train을 이렇게 정함으로써 episode마다 loss를 구한 횟수가 동일하게 된다. 
이를 통해 episodic loss가 일관된 기준 하에 구해진다.


```python
class DQN(Model):

    def __init__(self, action_n):
        super(DQN, self).__init__()

        self.h5 = Dense(224, activation='relu')
        self.h6 = Dense(112, activation='relu')
        self.h7 = Dense(56, activation='relu')
        self.h8 = Dense(28, activation='relu')
        self.q = Dense(action_n, activation='linear')


    def call(self, x):
        x = self.h5(x)
        x = self.h6(x)
        x = self.h7(x)
        x = self.h8(x)
        q = self.q(x)
        return q
```
Deepmind에서 ‘Playing Atari with Deep Reinforcement Learning’ 논문을 통해 atari game들을 play하는 강화학습 모델을 세상에 공개했는데, 여기서 5개의 layer를 사용했다고 한다. 
그래서 이 task를 해결하는 데도 5개의 layer면 충분하지 않을까 생각하고 시도해봤다. Unit 수는 deepmind 논문에서 다음 층으로 내려갈 때마다 반 또는 1/6로 줄어들게 만들어서, 여기서는 반으로 줄어들게 설정했다.
Call 메서드는 객체 자체를 호출할 때 순전파 결과를 반환한다.

 ```python
dqn = DQN(n_outputs)
target_dqn = DQN(n_outputs)

dqn.build(input_shape=X_shape)
target_dqn.build(input_shape=X_shape)
 ```
객체를 생성하고 .build()로 weight을 initialize한다. 

```python
def compute_td_target(rewards, target_qs):
  max_q = np.max(target_qs, axis=1, keepdims=True)
  y_k = np.zeros(max_q.shape)
  for i in range(max_q.shape[0]): # number of batch
    y_k[i] = rewards[i] + discount_factor * max_q[i]
  return y_k
```
Dqn의 loss를 구하기 위해, main network q값과의 차를 구할 때 쓰는 td_target을 계산하는 함수다.
Target network에서 구한 s(k+1)에서의 q값들인 target_qs에서 최댓값을 batch의 각 경우마다 고른다. 그 후 `y_k = r_k + gamma* max Q(s_k+1, a)` 형태로 td_target을 구한다.

```python
def dqn_learn(states, actions, td_targets):
  with tf.GradientTape() as tape:
      one_hot_actions = tf.one_hot(actions, n_outputs)
      q = dqn(states, training=True)
      q_values = tf.reduce_sum(one_hot_actions * q, axis=1, keepdims=True)
      loss = tf.reduce_mean(tf.square(q_values-td_targets))
  grads = tape.gradient(loss, dqn.trainable_variables)
  dqn_opt.apply_gradients(zip(grads, dqn.trainable_variables))
  return loss.numpy()
```
Loss를 구하고 이를 이용해 gradient를 구하고 역전파까지 실행하는 함수다.
q_values는 현 state에서 취한 action에 대한 main network의 q값이 batch마다 구해진 값이다.  이것과 td_targets의 차의 제곱을 loss로 삼아 학습한다.

```python
def test(test_reward_list):
  done = False
  # obs는 shape (23,) numpy array
  obs,_ = env.reset()
  episodic_reward = 0
  actions_counter = Counter()

  while not done:

    # actions shape (batch,14) numpy array
    actions = dqn(tf.convert_to_tensor([obs], dtype=tf.float32))

    # action shape (1,) numpy array
    action = np.argmax(actions, axis=-1)

    actions_counter[action[0]] += 1

    # action env에 맞춰 list로 변환
    action_env=change_action_format(action)

    next_obs,reward,terminated,truncated,_ = env.step(action_env)

    done= terminated or truncated

    obs = next_obs
    episodic_reward += reward

  print('gloabl step', global_step, 'Test Reward', episodic_reward)
  print('Test', actions_counter)
  test_reward_list.append(episodic_reward)
```
test 함수는 훈련을 하다가 20 episodes마다 한 번씩 실행되는 함수로, epsilon-greedy를 적용하지 않고 온전히 main network의 policy를 따라 env를 1 episode 도는 함수다. 말 그대로 현재 network의 능력을 확인해보는 함수다. 
episodic_reward로 episode에서 받은 총 reward를 계산하고, 이를 test_reward_list에 추가한다. test_reward_list에는 여태까지 실행한 test들에서의 episodic reward 결과가 차례대로 저장돼있다.
actions_counter는 episode를 돌 동안 0~13의 14가지 action을 각각 몇 번 실행했는지를 저장하는 counter다. 결과를 test가 종료되면 print한다.
Counter({2: 25, 0: 24, 1: 15, 6: 12, 5: 9, 7: 7, 13: 5, 8: 2, 9: 1})의 형태로 출력된다. 2:25는 2번 action을 25번 실행했다는 의미다.


```python
def test_and_render(num_trial):
  _display = Display(visible=False, size=(1400, 900))
  _ = _display.start()

  # 아래 기술된 대로 폴더를 준비해둬야 에러 발생x (현 디렉토리 맨 위 셀 참고)
  env1 = RecordVideo(env, video_folder="./비디오/newcamera_target_network_5layer")

  for i in range(num_trial):
    r=0
    state,_=env1.reset()
    actions_counter = Counter()

    while True:
      env1.render()

      actions = dqn(tf.convert_to_tensor([state], dtype=tf.float32))
      action = np.argmax(actions, axis=-1)
      action_env=change_action_format(action)

      actions_counter[action[0]] += 1


      nextstate,reward,terminated,truncated,info = env1.step(action_env)

      state=nextstate
      r+=reward

      done=terminated or truncated

      if done:
          print(f'total reward of the episode : {r:6.3f}')
          print(actions_counter)
          break
  env1.close()
  _display.stop()
```
test_and_render 함수는 test함수에 render 요소들을 추가한 함수다. 훈련을 하다가 현재 agent가 어느 정도의 수행 능력을 보이고 있는지 눈으로 보고 싶을 때 가끔 실행한 함수다.
렌더링할 화면이 요구되기 때문에 Display 클래스로 가상 디스플레이를 만들었다. 그리고 env에 RecordVideo wrapper를 씌워 명시된 폴더에 녹화한 video를 저장했다.
RecordVideo wrapper의 인자로 있는 video_folder에 해당하는 폴더를 미리 만들어놔야 에러가 발생하지 않는다. 

```python
# 불러오기 (새로운 모델 학습시킬 때 주석 처리)
# 아래 기술된 대로 폴더와 파일을 준비해둬야 에러 발생x (현 디렉토리 맨 위 셀 참고)
with open('./피클/newcamera_target_network_5layer/test_reward_list.p','rb') as f:
  test_reward_list=pickle.load(f)
with open('./피클/newcamera_target_network_5layer/episodic_loss_list.p','rb') as f:
  episodic_loss_list=pickle.load(f)

dqn.load_weights('./모델 weights/newcamera_target_network_5layer/save_weights')

# 초기화 (새로운 모델 학습시킬 때 주석 해제)
'''test_reward_list=[ ]
episodic_loss_list=[]'''
```
저장해둔 test_reward_list, episodic_loss_list, main_network의 weights를 불러오는 코드다. 코드 맨 처음에 change directory를 해서 현재 directory가 DQN 폴더인 상태이므로, DQN 폴더 아래에, 위에 기술된 대로 폴더들을 만들고 거기에 해당하는 파일을 넣어놔야 정상적으로 실행된다. (폴더 안에 만들어져 있습니다.)

```python
# 불러오기 (새로운 모델 학습시킬 때 주석 처리)
# 아래 기술된 대로 폴더와 파일을 준비해둬야 에러 발생x (현 디렉토리 맨 위 셀 참고)
'''with open('./피클/newcamera_target_network_5layer/test_reward_list.p','rb') as f:
  test_reward_list=pickle.load(f)
with open('./피클/newcamera_target_network_5layer/episodic_loss_list.p','rb') as f:
  episodic_loss_list=pickle.load(f)

dqn.load_weights('./모델 weights/newcamera_target_network_5layer/save_weights')'''

# 초기화 (새로운 모델 학습시킬 때 주석 해제)
test_reward_list=[ ]
episodic_loss_list=[]
```
새로운 dqn 모델에 대해 학습을 처음 시작할 때는 위와 같이 주석처리를 반대로 하여 실행한다. Dqn은 이미 위에서 초기화하였고, 여기서 나머지 변수들도 초기화한다.

```python
# 한번에 학습시키고 싶은 만큼 유동적으로
num_episodes = 30

global_step=0

# for each episode
for i in range(num_episodes):

  if i%20 == 0:
    # train 중에는 epsilon-greedy 때문에 reward가 능력에 비해 더 떨어질 수밖에 없음
    # 그래서 epsilon-greedy 없는 상태로 reward를 구해봄
    test(test_reward_list)

  done = False
  # obs는 shape (23,) numpy array
  obs,_ = env.reset()
  epoch = 0
  episodic_loss = 0
```
지금까지 함수, 변수들을 준비하는 과정이였고, 여기서부터 학습이 시작된다.
 num_episodes는 훈련하는 동안 돌 episodes의 횟수로, 실행시킬 시간적 여유가 얼마나 되는지에 따라 유동적으로 가져갔다.
20 episodes마다 한 번씩 test()를 수행했다.
Episode 시작이므로 관련 변수들을 reset 한다.


```python
 while not done:

    # actions shape (batch,14) numpy array
    actions = dqn(tf.convert_to_tensor([obs], dtype=tf.float32))
    # action shape (1,) numpy array
    action = np.argmax(actions.numpy(), axis=-1)

    action = epsilon_greedy(action, global_step)

    # action을 env에 맞춰 shape (7,)로 변환
    action_env=change_action_format(action)

    next_obs,reward,terminated,truncated,info = env.step(action_env)

    done= terminated or truncated

    exp_buffer.append([obs, action, next_obs, reward, done])
```
한 episode 간 수행되는 루프 안이다. dqn의 call 메서드를 호출해 state에서의 q값들을 구한다. 여기서 최댓값을 갖는 action을 구하고 epsilon-greedy로 랜덤성을 부여한다.  Env가 요구하는 형태로 action을 (7,) shape으로 변환하고 env.step을 밟는다. 나온 결과를 experience replay buffer에 추가한다. 

```python
    if global_step % steps_train == 0 and global_step >= start_steps:

      o_obs, o_act, o_next_obs, o_rew, o_done = sample_memories(batch_size)

      #o_obs와 o_next_obs 의 shape이 (batch,) 이고 numpy array가 원소로 인식돼있는데 이걸 (batch,23)으로 만들어야
      o_obs=np.concatenate([a for a in o_obs]).reshape(-1,23)
      o_next_obs=np.concatenate([a for a in o_next_obs]).reshape(-1,23)

      # next state에서의 target Q-values
      target_qs = target_dqn(tf.convert_to_tensor(o_next_obs, dtype=tf.float32))
      td_target = compute_td_target(o_rew, target_qs.numpy())

      loss=dqn_learn(tf.convert_to_tensor(o_obs, dtype=tf.float32),
                      o_act,
                      tf.convert_to_tensor(td_target, dtype=tf.float32))
      episodic_loss += loss
```
학습이 일어나는 구간이다. batch_size만큼의 <s,a,r,s’>을 experience replay에서 가져온다. sample_memories 함수에서 dtype=object로 하여 o_obs와 o_next_obs를 가져왔기 때문에 (23,) shape의 각 원소들은 포인터 형태로 가져와진 상태다. 따라서 (batch,) shape으로 돼있다. 그런데, 그 아래 있는 target_dqn()과 dqn_learn()에서는 이들을 (batch,23) shape으로 가정하고 만들었다. 따라서 (batch,)를 (batch,23)으로 변환해줘야 한다. 이를 포인터로 된 각 원소에서 값을 가져와서 np.concatenate로 연결하고 reshape하는 방식으로 해결했다.
target_dqn의 call 메서드를 통해 next state에 대한 target network의 q값들을 뽑는다. 이를 이용해 `td target y_k = r_k + gamma* max Q(s_k+1, a)`을 구한다. 
그리고 이를 이용해 loss를 구하고 backpropagation까지 진행한다.
이때 구해진 loss는 episodic loss에 더해준다.

```python
    if global_step % copy_steps == 0 and global_step >= start_steps:
      update_target_network()
```
copy_steps마다 main network를 target network로 복사해온다. 

```python
  # 아직 학습이 시작되지 않았을 때는 리스트에 추가하지 않고자
  if global_step > start_steps:
    episodic_loss_list.append(episodic_loss)

```
global_step이 start_step이하일 때는 아직 학습이 한 번도 진행되지 않은 상태다. 따라서 episodic_loss는 그냥 0이다. 이 값이 episodic_loss_list에 포함되면 추후에 그래프로 나타낼 때 학습이 잘 된 것처럼 보이는 오해의 소지가 있으므로 리스트에 추가되지 않도록 한다. 

```python
  if (i+1)%period_print_result==0:
    # loss는 최근 period_print_result개 episode 평균값
    # 처음에 episode_to_start_train-1 만큼의 episode 동안은 episodic loss를 구하지 않으므로 이때만 리스트 슬라이싱 범위를 줄임
    if (i+1)//period_print_result==1:
      print('Loss_avg', sum(episodic_loss_list[-1:-1-period_print_result+(episode_to_start_train-1):-1])/(period_print_result-(episode_to_start_train-1)))
    else:
      print('Loss_avg', sum(episodic_loss_list[-1:-1-period_print_result:-1])/period_print_result)
```
코드가 실행되는 도중에 현재 loss가 어떤 추세를 보이고 있는지 확인하기 위한 코드다. period_print_result(위에서 10으로 지정) episode마다 실행된다. 처음 10번 episode를 돌았을 때는 첫 4번의 episode에서 episodic loss를 구하지 않았고 리스트에 추가하지도 않았기 때문에 6개 episode의 값만을 평균내야 한다. 이를 코드로 나타냈다. 나머지 경우들에서는 최근 10개 episodic loss값을 평균내 출력한다. 

```python
dqn.save_weights("./모델 weights/newcamera_target_network_5layer/save_weights")
```
num_episodes만큼의 학습이 끝난 이후 dqn 모델 weights를 해당 폴더에 저장하고 학습을 마친다.

```python
# 오늘 다 돌려서 저장해둘 때
with open('./피클/newcamera_target_network_5layer/test_reward_list.p','wb') as f:
  pickle.dump(test_reward_list,f)
with open('./피클/newcamera_target_network_5layer/episodic_loss_list.p','wb') as f:
  pickle.dump(episodic_loss_list,f)
```
test_reward_list와 episodic_loss_list도 차후에 이어서 결과를 추가하고자 폴더에 피클 객체로 저장해둔다. 

```python
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(20,5))
ax1.plot(test_reward_list,'o:', markersize=3, color='blue', linewidth=1)
ax2.plot(episodic_loss_list,'o:', markersize=3, color='blue', linewidth=1)

ax1.set_title('test_reward')
ax2.set_title('episodic_loss')
ax1.set_xlabel('num of entire episodes / 20')
ax1.set_ylabel('reward')
ax2.set_xlabel('num of trained episodes')
ax2.set_ylabel('loss')

plt.show()
```
test_reward_list와 episodic_loss_list를 그래프로 나타내는 코드다. 각 데이터를 점으로 나타냈다. Test는 20 episodes마다 1번씩 진행했고, episodic loss는 학습이 진행된 매 episodes마다 구해졌다. 따라서 episodic loss list 그래프의 x축은 훈련한 episodes 수와 같다. 그리고 test_reward_list 그래프의 x축은 {전체 episdoes 수}/20과 같다. 

```python
# 한 번에 동영상 2개가 최대, 한 세션 안에서 함수 다시 실행시키면 에러는 안 나는데 동영상 안 만들어짐 더 보고 싶으면 세션 다시 시작 해야
test_and_render(2)
```
학습을 하다가, 지금 agent가 어느 정도의 수행 능력을 보이고 있나 동영상 형태로 확인하고 싶을 때 실행했다. 
실행시 주의할 점은, 인자로 3이상을 줘도, 즉 한 번에 3개 이상의 동영상을 만들고자 해도 2개의 동영상 밖에 만들어지지 않는다.  그래서 인자를 2로 사용했다. 
또한 한 세션 안에서 이 함수를 또 실행시키면, 에러는 나지 않지만 해당 폴더에 동영상은 만들어져 있지 않는다. 그래서 10개의 동영상을 뽑아보고 싶으면, ‘세션 다시 시작’을 4번 추가로 해야 한다. 
프로젝트에서 결과를 내는 데 있어 critical한 문제는 아니었고 자주 render할 일도 없었기 때문에 굳이 해결하고자 알아보지는 않았다.

```python
# 위에서 녹화한 결과를 띄움
# 두번째 동영상을 보고 싶으면 video_path의 0을 1로

from IPython.display import HTML
from base64 import b64encode
# episode 0 또는 1로
video_path = '비디오/newcamera_target_network_5layer/rl-video-episode-1.mp4'

mp4 = open(video_path,'rb').read()
decoded_vid = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f'<video width=400 controls><source src={decoded_vid} type="video/mp4"></video>')
```
test_and_render()로 만들어진 동영상을 주피터 노트북 내에서 재생해볼 수 있도록 하는 코드다. 비디오가 만들어질 때 첫번째 동영상은 rl-video-episode-0, 두번째 동영상은 rl-video-episode-1으로 만들어진다. 두번째 동영상을 보고 싶으면 video_path의 0을 1로 바꾸면 된다. 
또는 그냥 해당 폴더로 가서 mp4파일을 열어봐도 된다. 



## [코드 실행 결과 및 물리엔진 mujoco의 버그](#목차)
이제 코드를 실행해봤다.
epsilon을 1\~0.8로 높게 시작해서 어느 정도 안정화됐을 때(episode 8000쯤) 0.5\~0.3으로 낮춰 학습시켜봤다.
결과는 아래와 같다.

![image](https://github.com/user-attachments/assets/6ea7afd9-0c57-48e0-a25b-2eb7e1756f8c)
![image](https://github.com/user-attachments/assets/53384007-b9f6-489f-a197-d9b14872ee08)
 
Test reward가 개선되다가 (-30 후반대)\~(-50중반대)에서 횡보하는 형태로 나타난다.
학습이 어느 정도 진행된 것 같아 test_and_render 함수를 통해 결과를 동영상으로 확인해봤다. 
Agent가 물체 주변에 접근하는 것까지 잘 학습했다는 걸 확인할 수 있었다. 
그런데, 동영상에서 충격적인 사실을 마주한다.
로봇의 fingertip이 물건을 뚫고 가는 버그가 있는 것이다. (아래 동영상)

인터넷에 검색을 해서 https://github.com/Farama-Foundation/Gymnasium/issues/950를 찾았다. 주요 내용은 다음과 같다.

위에 기술된 것처럼, 비디오를 찍는 카메라 각도와 고도를 바꾸고, mujoco 3.1.2가 아닌 2.3.7 버전을 쓸 때 에러가 해결됐다.
이 내용을 코드에 적용시켰다.

이후 비디오는 이런 각도가 됐다.


## [버그 해결 후 재학습 결과](#목차)
### 버그 있는 채로 학습했던 모델에 이어서 학습시킨 결과
env를 바꾼 후 episode 16000부터 같은 모델에 이어서 학습시켰더니
env가 달라진 만큼 그래프 상으로 확실한 변화가 나타났다.
아래 두 그래프 중 Test_reward 800(=16000/20)이후를 보면 지금까지의 최고 결과였던 -30후반대를 뛰어넘는 좋은 모습이 나오는 한편, 안 좋을 때는 더 안 좋아지는 경우들도 생기며 편차가 커졌다. 
학습이 진행됨에 따라 episodic loss는 여태까지와는 비교도 안 될 만큼 커지고, test_reward 결과도 악화돼갔다. 여태까지 학습한 게 있는데 갑자기 새로운 게 섞이다보니 충분히 그럴 만하다고 생각한다.
 

그래서 아예 초기화된 weight로 새롭게 다시 시작했다.

### 새롭게 학습시킨 결과
test reward가 -30대 위주로 나오는데, 이게 어느 정도 퍼포먼스를 보이는 건지 궁금해서 test_and_render를 해봤다. 그 결과, -30대는 성공적으로 task를 수행했을 때의 결과였다. test_and_render를 통해 20개 이상의 동영상을 뽑아봤는데, -30대가 아닌 것은 -47.906, -41.520으로 두 번 나왔다. 아래는 뽑은 동영상들의 일부다. (뽑아본 동영상 전부는 DQN/비디오 폴더 내부에 있습니다.)
    
-33.007					   -32.366
    
-30.037					   -36.092

-33.671					   -31.826


안 좋게 나온 것들을 살펴보면,
한 번 -47.906이 나왔는데, 물체가 로봇 몸 가까이에 있을 때 팔을 뒤로 덜 빼고, 팔로 실린더를 위에서 누르듯이 건들기 시작하다보니, 실린더가 이상한 방향으로 튀어나가면서 발생한 문제였다.
두 번째로 안 좋았던 -41.520도, 같은 원인이었다. 물체가 가까이 있어 fingertip이 아닌 팔로 건들기 시작했다. 이로 인해 물체가 약간 오른쪽으로 빠져나가면서 fingertip 범위 밖으로 나갔다. 그러나 agent는 물체가 fingertip 사이에 있다는 생각으로 행동을 이어가면서 episode가 마무리됐다.
 

Reward -33.860, -34.206이었던 경우로, 위와 비슷한 식으로 건드려서 위험해 보이지만 성공한 경우도 있었다.
     



reward -39.301으로, 팔로 건들기 시작했지만 나쁘지 않은 경우도 없진 않다.

## [결과 부족한 부분 해결책](#목차)
팔로 건드려서 발생하는 이 문제는, 엄밀히 말하면 팔로 건드려서라기 보다는 물체 윗면을 건드려서 발생하는 문제다. 왜냐면 fingertip으로 위에서 누르듯이 물체 윗면을 건드려도 똑 같은 문제가 발생할 것이기 때문이다.
따라서 이를 해결하려면,
Agent가 물체와 처음 접촉할 때 그 접촉 높이가 물체 윗면 높이에 해당할 때, reward로 큰 페널티를 부과하면 될 것이다.

이를 구현하려면,
매 step마다 접촉 시작했는지 확인하고, 조건을 만족할 때 페널티를 부과해야하므로 step()이 내부적으로 어떻게 작동하는지 알아야한다.
 
깃허브의 gym 레포지토리로 가서 살펴봤더니, step()은
`pusher_v4.py`의 PusherEnv 클래스 내에 있었다. 
step() 안에서 핵심 구현을 하는 `self.get_body_com("object")`과 `self.do_simulation(a, self.frame_skip)`는 해당 클래스가 상속한 MujocoEnv에서 온 함수다.

`mujoco_env.py`로 가보면, 
`self.get_body_com("object")`는 `return self.data.body(body_name).xpos`가 그 내용

여기서 나온 `self.data`는 같은 파일의 `_initialize_simulation()`에서  정의 
 
`self.model = mujoco.MjModel.from_xml_path(self.fullpath)`
`self.data = mujoco.MjData(self.model)`
결국 xml(xml은 `mujoco/assets/pusher.xml`에 있음)로 정리된 환경에 대한 정보들을 mujoco에서 받아들일 수 있는 형태로 정리한 게 `self.data`고, `get_body_com()`은 여기서 그 정보를 추출함으로써 구해진 거다.

위에서알아보려고 한 두번째 함수 `self.do_simulation(a, self.frame_skip)`는 `self._step_mujoco_simulation(ctrl, n_frames)`를 호출

이 함수는 다시 `self.data.ctrl[:] = ctrl`,  `mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)` 가 내용. 
결국, `mujoco.mj_step`을 통해 mujoco로 환경에 대한 정보를 보내 simulation을 외주하는 느낌이다.

정리하면, xml로 환경에 대한 정보들을 정리해두고 있고, simulation이 필요할 때는 이를  mujoco쪽의 함수에 제공하면, mujoco에서 simulation을 하고 거기서 결과를 가져오는 방식으로 진행된다는 걸 알 수 있다.

따라서 mujoco 쪽에서 접촉에 대한 정보를 주는 함수가 있나 알아보고, 이것과 xml의 변수를 이용해 step() 함수에서 reward를 추가해야 할 것이다.

copilot에게 물어본 결과,
```python
# Function to check if two objects are in contact and get the height of the collision
def get_collision_height(data, geom1, geom2):
    for contact in data.contact[:data.ncon]:
        if (contact.geom1 == geom1 and contact.geom2 == geom2) or (contact.geom1 == geom2 and contact.geom2 == geom1):
            # The contact position is stored in the `pos` attribute (x, y, z)
            return contact.pos[2]  # z-coordinate
    return None

def main():
    # Load the model and create a simulation
    model = mujoco_py.load_model_from_path('path/to/your/model.xml')
    sim = mujoco_py.MjSim(model)

    # IDs of the geometries to check for contact
    geom1 = model.geom_name2id('geom1_name')
    geom2 = model.geom_name2id('geom2_name')

    while True:
        sim.step()
        collision_height = get_collision_height(sim.data, geom1, geom2)
```

다음과 같이 collison_height을 구할 수 있을 것으로 보인다.
따라서, step()에서 이 내용을 이용해 reward를 추가하면 충분히 구현할 수 있을 것 같다.
그러나 시간 상의 이유로 구현까지 할 수는 없었다.
하지만 구현해보고 싶은 사람이 있다면, 위 내용을 참고해 구현하면 될 것 같다.

## [기타 흥미로운 내용](#목차)
추가적인 내용인데, 보고서 작성 중 pusher에 대한 설명 페이지가 gymnasium v0.29.0용 말고 다른 버전용 페이지도 있다는 걸 찾게 됐다. 여기서 강화학습적으로 흥미로운 내용이 있었다.
  
(Pusher 홈페이지 오른쪽 하단에서 선택할 수 있다.)
Gymnasium v1.0.0a1의 pusher-v5에 대한 설명 중 이런 내용이 있었다.
 
env.step이 리턴하는 reward가 이번 action에 대한 reward가 아닌 직전 action에 대한 reward였다는 것이다. 내가 사용한 pusher-v4까지는 이랬다고 한다.
캡처에 파란 글씨로 써있는 GitHub issue 하이퍼링크로 들어가 봤다.
이 버그를 고친 사람이 TD3 알고리즘(Twin Delayed Deep Deterministic policy gradient algorithm)을 이용해 버그를 고치기 전후 상태에서 episodic returns를 구해보고 결과를 올려놨다.
  
결과를 보면 별 차이가 없다는 걸 알 수 있다. 그래도 바로 붙어있는 두 state 사이에서 일어난 일이다보니 결과에 큰 차이를 가져오진 않았다고 판단된다.
이렇게 해도 결과가 좋게 나온다는 게 흥미로웠다.


## [느낀 점](#목차)
이번 프로젝트에서 가장 기억에 남는 점은, 처음에 q-learning을 선택해 코드도 다 짰는데, 실행했더니 number of visted states가 끊임없이 선형적으로 증가하는 출력 결과를 보며, ‘아! 공간복잡도가 엄청 크겠구나!’ 하고 머리를 띵 얻어맞은 듯한 느낌이 들었던 것이다. 앞으로는 이런 멍청한 행동을 하지 않기 위해서라도 어떤 알고리즘을 선택할 때 알아서 시간복잡도와 공간복잡도를 따져보고 시작할 것 같다. 
그리고 dqn을 이용해 성공적으로 학습된 걸 동영상으로 처음 봤을 때 느낀 희열이 기억에 남는다. 
마지막으로, 차후 다양한 문제에도 강화학습을 활용할 수 있겠다는 자신감을 얻게 됐다.

그런데 한편으로는, 강화학습의 발전성에 대한 회의감도 들었다.
앞에서 agent가 물체 윗부분을 건드려서 task를 잘 수행하지 못하는 경우를 개선하는 것과 같이, agent가 특정 상황에서 문제를 보이면, reward 제공 규칙을 새로 작성해줘야 한다.
그러나 인간의 경우, '물체 윗부분을 건들지 않게 하는 게 좋을 것 같아'라고 조언하면, 물체 윗부분 쪽으로 갈 때 그 조언을 연상하고 하려던 행동을 억제할 줄 안다.
따라서 무작위에서 시작해서 reward에 따라 학습해나가는 강화학습과, 언어를 토대로 학습해나가는 transformer 방식이 결합돼야 보다 인간스러운 방식으로 학습하는 로봇이 탄생할 수 있겠다는 생각이 들었다.

## [참고자료](#목차)
- gymnasium v0.29.0 pusher documentation : https://gymnasium.farama.org/v0.29.0/environments/mujoco/pusher/
- gymnasium basic usage & wrappers/misc wrappers/RecordVideo documentation : 위 사이트 왼쪽 배너 참고
- CartPole-v1 환경 DQN : https://pasus.tistory.com/133
- 코랩에서 가상 디스플레이 생성 : https://jellyho.com/blog/102/
- 코랩에서 동영상 재생 : https://blog.naver.com/baemsu/223198658162
