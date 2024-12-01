# Training_Pusher_by_DQN

## 목차


## 실행 방법
환경 : colab을 이용해주시면 됩니다. 설치해야할 패키지들은 모두 코드로 포함돼있습니다.
과제 제출시 제출한 ‘3-2 강화학습 프로젝트’ 압축 폴더를 푸시고, ‘3-2 강화학습 프로젝트’ 폴더 그대로(colab에서 코드상 에러가 나지 않으려면 내부 디렉토리를 건들면 안 됩니다) 구글 드라이브의 ‘내 드라이브’에 업로드하시면 됩니다. 코드도 폴더에 포함돼있습니다. q-learning.ipynb와 DQN.ipynb입니다.

# 프로젝트 개요
gymnasium의 pusher
Pusher는 위 동영상과 같이 다관절 로봇이 흰색 물체를 빨간색 지점으로 옮기는 게 목표인 환경이다.


## pusher 환경 설명

### action
![image](https://github.com/user-attachments/assets/538bf958-09c2-443e-8529-46815200f52d)
Action Space는 shape이 (7,), (최솟값,최댓값)은 (-2.0, 2.0), 데이터형은 float32이다.
action의 종류는 왼쪽 캡처와 같다.
0의 경우, 어깨를 이용해 팔을 좌우로 움직이는 것, 1은 어깨를 이용해 팔을 올리고 내리는 것, 2는 어깨를 돌리는 것, 3은 팔꿈치를 기준으로 전완을 굽히고 펴는 것, 4는 팔꿈치를 기준으로 전완을 돌리는 것, 5는 손목을 굽히고 펴는 것, 6은 손목을 돌리는 걸 가리킨다.

![image](https://github.com/user-attachments/assets/3c3d8671-50a7-4561-9df8-959c692372f0)
Observation Space는 shape이 (23,), (최솟값,최댓값)은 (-inf, inf), 데이터형은 float64다,
Observation의 0~6은 현재 joint들이 얼마나 회전된 상태인지를 나타낸다. 여기서 0~6이 맡는 joint는 action 0~6이 의미하는 joint와 순서가 일치한다.
Observation의 7~13은 joint들의 각속도를 나타내며, 마찬가지로 7~13이 맡는 joint 순서는 action 0~6의 joint 순서와 같다.
14~16은 로봇의 손가락 위치의 x,y,z 좌표다.
17~19는 옮겨야할 물건 위치의 x,y,z 좌표다.
20~22는 물건이 옮겨져야할 목표 위치의 x,y,z 좌표다.

![image](https://github.com/user-attachments/assets/caf5ec43-622b-401c-8679-3b580d165ad8)
Rewards는 세 가지로 이뤄져 있다. 
1. reward_near : -(로봇의 fingertip과 목표지점의 거리)
2. reward_dist : -(물체와 목표지점의 거리)
3. reward_control : -(action의 크기) 로, 7가지 action을 각각 제곱해서 더함으로써 action의 크기를 구함.
그리고 전체 reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near 와 같이 가중치를 부여하여 각각을 합친다.
Reward의 모든 요소는 로봇이 얼마나 못하고 있는지를 나타내는 형태로, 항상 음수이며, 매 step마다 무조건 reward를 부여하는 환경임을 알 수 있다.
그리고 reward 요소들의 내용을 보면, 매순간순간 agent가 수행을 얼마나 잘하고 있는지, 직전에 비해 더 나은 방향으로 가고 있는지를 잘 알려줌을 알 수 있다.

![image](https://github.com/user-attachments/assets/66f65478-bd3a-4c81-8736-30e6f0a5f32e)
환경이 시작될 때, 로봇 관절들의 각변위는 항상 0이며, 각속도는 0을 기준으로 uniform noise [-0.005, 0.005]를 갖고 시작한다.
물체의 x좌표는 [-0.3, 0]에서, y좌표는 [-0.2, 0.2]에서 랜덤하게 정해진다. 이때 (x,y)의 원점과의 거리가 0.17이 넘지 않게 정해진다.
목표지점의 좌표는 (0.45, -0.05, -0.323)로 고정이다.
한 번의 env.step(action)은 5프레임 간 지속된다. 1프레임은 0.01초이므로 한 action은 0.05초 간 지속된다. 

![image](https://github.com/user-attachments/assets/e6372a0e-98dc-47c9-8bba-cc8e784adbc2)
첫 페이지에서 언급했듯 gymnasium의 gym과의 차이점은, done 대신 truncation과 termination 두 개를 사용하는 거다. 원래의 done과 같은 개념이 termination이다. 
Truncation은 한 episode가 지지부진하게 이어져서 시간만 낭비되는 걸 막고자 일정한 시간이 되면 episode를 그냥 끝내버리는 거다. 이 environment의 경우, 100번째 env.step이 끝나고 truncated 된다. 직전 페이지에서 한 step은 0.05초 지속된다 했으므로, 이 environment는 truncated 될 경우, 한 episode가 5초 동안 진행된다는 걸 알 수 있다.
Termination은 state space value가 infinite해질 때 된다고 써있다.
그런데 이 환경에서 action을 처음부터 끝까지 최대 크기로 해서 실행해본 결과, truncated 되는 걸 확인했다. Truncated 기준이 5초일 때는 state가 infinite해질 일이 없는 것이다. 즉, 이번 실습 상에서는 terminated 될 일이 없다. 항상 truncated 된다.


## Q-learning 시도
처음에는 Q-learning을 시도했다. 그러나 Q-learning으로는 불가능하다는 걸 깨닫고 DQN을 이용해서 성공적으로 학습시켰다.

Q-learning으로 코드를 다 짜고 실행을 했다. 실행을 했는데 number of visted (s,a)가 episode수에 따라 선형적으로 증가하고, 대부분의 state는 한번 방문한 것으로 떴다. 그리고 7만번 정도의 episode를 돌고 나니 ram부족으로 런타임이 종료됐다. 이때 깨달았다. 애초에 pusher env를 해결하기 위해 q-learning을 쓰면 안 됐다. 이 env는 State space, action space는 총 23+7=30 개의 인자들로 이뤄져 있다. 그리고 각 인자는 -inf~inf 또는 -2~2의 연속형 변수다. 그런데 각 인자에서 고작 10개의 discrete한 수준값만 시도한다고 해도 (s,a) 경우의 수는 10^30이 된다. 해당 경우들의 Q값만 저장해도 4*10^30 byte = 4*10^18 byte가 필요하다. 터무니 없는 수치다. Q-learning이 state, action이 복잡하면 사용할 수 없는 알고리즘이란 걸 당연히 알고는 있었지만, 그 말의 의미를 깊게 생각해보지 않은 결과였다. 앞으로는 알고리즘을 적용할 때 미리 공간복잡도를 꼭 따져봐야겠다는 다짐을 강하게 했다.

dqn의 위력을 실감하고, dqn을 이용해 새로운 마음으로 다시 시작했다.



## 코드 설명



## 코드 실행 결과 및 충격적인 mujoco 버그
epsilon을 1~0.8로 높게 시작해서 어느 정도 안정화됐을 때(episode 8000쯤) 0.5~0.3으로 낮춰 학습시켜봤다.
결과는 아래와 같다.
![image](https://github.com/user-attachments/assets/0ca9eec4-f3a0-4d89-ba3f-94576c217820)
![image](https://github.com/user-attachments/assets/cc952f21-1271-435b-a522-e74c2acd395a)
Test reward가 개선되다가 (-30 후반대)~(-50중반대)에서 횡보하는 형태로 나타난다.
학습이 어느 정도 진행된 것 같아 test_and_render 함수를 통해 결과를 동영상으로 확인해봤다. 
Agent가 물체 주변에 접근하는 것까지 잘 학습했다는 걸 확인할 수 있었다. 
그런데, 동영상에서 충격적인 사실을 마주한다.
로봇의 fingertip이 물건을 뚫고 가는 버그가 있는 것이다. (아래 동영상)

## 버그 해결 후 재학습 결과

## 기타 흥미로운 내용

## 느낀 점
