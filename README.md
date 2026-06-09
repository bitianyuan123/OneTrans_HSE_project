# OneTrans

### Структура репозитория

onetrans/
- models/ - основные модули для бейзлайнов и OneTrans
- ext/ - все нужное для использования Yambda датасета
- nn/ - все модули нейросетей
- - attention/ - различные реализации attention блоков
- - blocks/ - различные агрегации слоев в блоки 
- - encoders/ - кодировщики данных
- - ffn/ - различные реализации ffn блоков
- - tokenizer.py - токенизаторы для OneTrans
- run/ - папка для запуска экспериментов в модели OneTrans
- - main.py - запуск OneTrans
- run_baselines/ - папка для запуска экспериментов с бейзлайнами
- - run_catboost.py - запуск Catboost модели
- - run_dcn - запуск DCN модели
- - run_hiformer - запуск Hiformer модели
- utils/ - различные вспомогательные функции для работы модулей

### Инструкция для запуска экспериментов


Наш сетап для экспериментов это: python 3.14, Kaggle GPU T4, менеджер пакетов uv

#### Установите uv и все зависимости

```shell
pip install uv
uv sync
```

#### Запуск эксперимента

Можно напрямую запускать эксперимент через запуск файла в модуле *run_baselines* или *run*, 
но для удобства можно использовать следующий шаблонный скрипт для Kaggle, предварительно установив *WANDB_API_KEY* в окружение 

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("WANDB_API_KEY")

os.chdir("/kaggle/working")
os.system("pip install uv -q")
os.system("rm -rf /kaggle/working/OneTrans_HSE_project")
os.system("git clone https://github.com/olezha223/OneTrans_HSE_project -q")
os.chdir("/kaggle/working/OneTrans_HSE_project")
os.system("uv sync -q")

os.system(
    f"WANDB_API_KEY={wandb_api_key} "
    "uv run python -m onetrans.run_baselines.run_catboost "
    "--iterations=100 "
    "--verbose=10 "
    "--lr=0.1 "
    "--num_workers=0 "
    "--run_name=baseline-catboost "
)
```

Меняя аргументы при запуске, можно менять параметры модели и ее обучения.
