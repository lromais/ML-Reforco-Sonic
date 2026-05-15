# 🦔 Sonic RL — Aprendizado por Reforço com PPO

Agente de aprendizado por reforço treinado para jogar **Sonic The Hedgehog (Genesis)** usando o algoritmo PPO (Proximal Policy Optimization) com a biblioteca Stable-Baselines3.


---

## 📋 Requisitos

- Python 3.10
- GPU NVIDIA com CUDA (recomendado, mas roda em CPU também)
- Driver NVIDIA atualizado (535+)
- ROM do jogo Sonic The Hedgehog (Genesis)

---

## ⚙️ Instalação

### 1. Instalar o pyenv e configurar o ambiente

```bash
# Instalar pyenv
curl https://pyenv.run | bash

# Adicionar ao ~/.bashrc ou ~/.zshrc:
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Recarregar shell e instalar Python 3.10
pyenv install 3.10.14
pyenv local 3.10.14  # define 3.10 pra pasta atual

# Criar venv
python -m venv venv_sonic
source venv_sonic/bin/activate
```

### 2. Instalar dependências do sistema (se necessário)

```bash
sudo apt install git cmake zlib1g-dev
```

### 3. Instalar as dependências Python

```bash
pip install --upgrade pip
pip install setuptools==67.8.0
pip install git+https://github.com/Farama-Foundation/stable-retro.git
pip install stable-baselines3[extra] gymnasium opencv-python pygame
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

> ⚠️ O `gym-retro` original está abandonado e quebra com versões novas do setuptools. Use o **stable-retro** instalado direto do GitHub como acima.

> ⚠️ O PyTorch deve ser instalado com suporte a CUDA (`cu121`). Isso garante que a GPU seja usada durante o treino.

### 4. Confirmar que a GPU está disponível

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Deve retornar `True`. Se retornar `False`, verifique se o driver NVIDIA está atualizado:

```bash
nvidia-smi
```

### 5. Importar a ROM do Sonic

Coloque a ROM (arquivo `.md`) dentro de uma pasta `roms/` e copie-a diretamente para o diretório do stable-retro:

```bash
python -c "
import stable_retro, shutil, os
game_path = os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable', 'SonicTheHedgehog-Genesis-v0')
shutil.copy('roms/Sonic The Hedgehog (USA, Europe).md', os.path.join(game_path, 'rom.md'))
print('ROM copiada para:', game_path)
"
```

> ⚠️ O nome da pasta no stable-retro é `SonicTheHedgehog-Genesis-v0` (com `-v0` no final). Para confirmar os nomes disponíveis no seu sistema:
> ```bash
> python -c "
> import stable_retro, os
> data_path = os.path.join(os.path.dirname(stable_retro.__file__), 'data', 'stable')
> print([j for j in os.listdir(data_path) if 'Sonic' in j])
> "
> ```

---

## 🚀 Como usar

O projeto é dividido em dois scripts: um para treino e outro para teste com visualização.

### Treinar o agente

```bash
python sonic_train.py
```

O treino roda **sem janela gráfica** (`render_mode=None`) para máxima performance. O terminal deve exibir `Usando device: cuda`, confirmando que a GPU está sendo usada.

O treinamento roda por **1.000.000 de timesteps** e salva o modelo ao final como `sonic_ppo.zip`.

### Acompanhar o treino com TensorBoard

Enquanto o treino roda, abra outro terminal e execute:

```bash
source venv_sonic/bin/activate
tensorboard --logdir ./logs/
```

Acesse `http://localhost:6006` no navegador para ver em tempo real:

- **Reward médio por episódio** — sobe conforme o Sonic aprende
- **Episode length** — episódios mais longos = Sonic sobrevivendo mais
- **Loss, entropy** — métricas internas de aprendizado

> O PPO coleta 2048 frames antes de cada atualização, então a janela do jogo ficaria travada a maior parte do tempo durante o treino. Por isso o TensorBoard é a melhor forma de acompanhar o progresso.

### Testar o modelo treinado

Após o treino, rode:

```bash
python sonic_test.py
```

Isso carrega o modelo salvo e abre uma janela **OpenCV** exibindo o jogo colorido em 640x480. Pressione `Q` para fechar.

---

## 🧠 Arquitetura

### Wrappers aplicados ao ambiente

| Wrapper | Descrição |
|---|---|
| `SkipFrame` | Repete cada ação por 4 frames (mais eficiente) |
| `GrayscaleObservation` | Converte frames para escala de cinza — sem `keep_dim` |
| `ResizeObservation` | Redimensiona frames para 84x84 pixels |
| `SonicReward` | Recompensa customizada (aplicada antes do FrameStack) |
| `FrameStackObservation` | Empilha 4 frames consecutivos — gera shape `(4, 84, 84)` |
| `TransposeObservation` | Garante o formato correto pro CnnPolicy |
| `Monitor` | Registra métricas de episódio |

### Recompensa customizada (`SonicReward`)

- **+1.0** por avançar para uma nova posição máxima no eixo X
- **-0.01** por ficar parado ou recuar
- **+0.001** bônus de sobrevivência a cada step

### Hiperparâmetros do PPO

| Parâmetro | Valor |
|---|---|
| `learning_rate` | 0.00025 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |

---

## 📁 Estrutura do projeto

```
.
├── sonic_train.py    # Script de treino (sem janela, máxima performance)
├── sonic_test.py     # Script de teste com visualização em tempo real
├── roms/             # Pasta com a ROM do Sonic (.md)
├── sonic_ppo.zip     # Modelo salvo (gerado após o treino)
├── logs/             # Logs do TensorBoard
└── README.md
```

---

## 📝 Observações

- O script usa `import stable_retro as retro` — o pacote `retro` original foi descontinuado.
- O script detecta automaticamente se há GPU disponível. Se não houver CUDA, roda em CPU (bem mais lento).
- O estado do jogo deve ser iniciado com `retro.State.NONE` — usar uma string como `'GreenHillZone.Act1'` faz o emulador ficar preso numa tela azul no Linux.
- O `GrayscaleObservation` deve ser usado **sem** `keep_dim=True` — com ele o `FrameStackObservation` gera shapes incompatíveis.
- O `SonicReward` deve ser aplicado **antes** do `FrameStackObservation` para não quebrar o empilhamento de frames.
- O modelo salvo pode ser carregado para continuar o treino ou apenas para testes.

### Carregar modelo salvo para continuar o treino

```python
from stable_baselines3 import PPO
model = PPO.load("sonic_ppo", env=env)
model.learn(total_timesteps=500_000)  # continua de onde parou
model.save("sonic_ppo")
```
