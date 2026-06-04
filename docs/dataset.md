## Класс `RankingDataset` – универсальный загрузчик для моделей ранжирования

### Назначение

Класс `RankingDataset` предназначен для централизованной загрузки и предобработки данных из датасета `yambda-50m-lag-features` (который содержит как агрегированные lag‑признаки, так и сырые последовательности взаимодействий). Он возвращает готовые PyTorch `Dataset` для пяти типов моделей:

- **catboost** – только не‑последовательные признаки (dense + sparse + multivalent) в плоском табличном виде;
- **dcn** (DCNv2) – аналогично, но с сохранением категориальных фич как `sparse_features` и мультивалентных как `multivalent_features`;
- **hiformer** – такой же набор, как для DCNv2, но может быть дополнен специальной токенизацией;
- **one_trans** – помимо не‑последовательных фич добавляет историю пользователя (S‑токены): item_id и сигналы взаимодействий за последние `seq_len` событий;
- **rank_mixer** – аналогично OneTrans, получает и NS‑признаки, и последовательности.

Класс инкапсулирует всю логику: загрузку данных, построение временных окон истории, разделение на train/test по времени (последние 30 дней по умолчанию), преобразование в тензоры и структурирование выходного словаря в формате, совместимом с соответствующими моделями.

### Конфигурация датасета

Конфигурация задаётся через датакласс `DatasetConfig`:

```python
@dataclass
class DatasetConfig:
    dataset_type: str = 'flat'
    dataset_size: str = '50m'
    interaction_name: str = 'multi_event'
    default_like_window_seconds: int = 24 * 60 * 60
    lag_seconds: int = 15 * 60
```

Параметры `lag_seconds` и `default_like_window_seconds` на данный момент не используются в загрузчике (оставлены для совместимости с оригинальным ДЗ 3). Основной датасет загружается с Hugging Face по репозиторию `matfu21/yambda-50m-lag-features` и содержит колонки:

- `uid`, `item_id`, `timestamp`
- `is_like`, `is_full_play`, `is_skip`
- 15 lag‑признаков (см. `DENSE_COLUMNS`)
- `artist_ids`, `album_ids` – мультивалентные категориальные признаки (списки)

### Глобальные константы признаков

Перед созданием датасета определены кортежи, описывающие состав признаков:

```python
DENSE_COLUMNS = (
    "user_lag_listen_cnt", "user_lag_like_cnt", "user_lag_full_play_cnt",
    "user_lag_skip_cnt", "item_lag_listen_cnt", "item_lag_like_cnt",
    "item_lag_full_play_cnt", "item_lag_skip_cnt", "ui_lag_listen_cnt",
    "ui_lag_like_cnt", "ui_lag_full_play_cnt", "ui_lag_skip_cnt",
    "user_lag_avg_played_ratio", "item_lag_avg_played_ratio",
    "ui_lag_avg_played_ratio",
)
MULTIVALENT_COLUMNS = ("artist_ids", "album_ids")
SPARSE_COLUMNS = ("uid", "item_id")
LABEL_COLUMNS = ("is_like", "is_full_play")
```

### Структура выходного датасета

Метод `get_dataset(model_name, split, seq_len, target, test_last_days)` возвращает экземпляр `_ModelSpecificDataset` – наследника `torch.utils.data.Dataset`. Каждый элемент этого датасета (один sample) – это словарь, формат которого зависит от модели:

#### Для всех моделей (общие поля)

- `"dense"` : `torch.Tensor` формы `(num_dense_features,)` типа `float32` – значения всех lag‑признаков текущего события.
- `"sparse"` : `torch.Tensor` формы `(2,)` типа `long` – `[uid, item_id]` текущего события.
- `"multivalent"` : словарь, где для каждого имени из `MULTIVALENT_COLUMNS` хранится вложенный словарь с ключами:
  - `"values"` : одномерный тензор `long` – конкатенация всех идентификаторов из списков `artist_ids` или `album_ids` для всех объектов в батче (в случае батчевой обработки) **или** для одного объекта (при индексации одного sample). Для удобства в датасете, возвращающем отдельные sample, `values` – это просто список/тензор идентификаторов для данного sample, а `lengths` – скаляр.
  - `"lengths"` : тензор `long` длины `batch_size` (или скаляр) с длинами соответствующих списков.
- `"label"` : скалярный тензор `float32` – значение таргета (по умолчанию `is_full_play`).

#### Дополнительно для моделей `one_trans` и `rank_mixer`

- `"seq_items"` : тензор `long` формы `(seq_len,)` – идентификаторы айтемов в истории пользователя (последние `seq_len` событий, упорядоченные от самых старых к самым новым). Отсутствующие позиции (если у пользователя меньше `seq_len` событий) заполнены нулём.
- `"seq_signals"` : тензор `float32` формы `(seq_len, 3)` – для каждого шага истории содержит три сигнала: `is_like`, `is_full_play`, `is_skip` (в указанном порядке). Padding заполнен нулями.
- `"seq_mask"` : булев тензор `(seq_len,)` – `True` для реальных событий истории, `False` для padding.

### Разделение train/test

Разделение выполняется на основе временной метки `timestamp`. Все события, у которых `timestamp` меньше чем `max_ts - test_last_days*24*3600`, попадают в тренировочную выборку, остальные – в тестовую. По умолчанию `test_last_days = 30`.

### Построение истории для OneTrans / RankMixer

История строится с помощью оконного сдвига (`shift(k).over("uid")`) для каждой строки датафрейма. Алгоритм:

1. Датафрейм сортируется по `uid`, затем по `timestamp`.
2. Для каждого `k` от 1 до `seq_len` создаются колонки:
   - `seq_item_{k}` – `item_id` из события, произошедшего на `k` шагов назад;
   - `seq_is_like_{k}`, `seq_is_full_play_{k}`, `seq_is_skip_{k}` – соответствующие сигналы.
3. Padding-значения (когда у пользователя нет `k`-го прошлого события) заменяются на `0` для item_id и сигналов.
4. Маска `seq_mask` вычисляется как `seq_items != 0`.

### Формат батчей: почему не батчевание внутри датасета?

В отличие от ДЗ 3, данный `RankingDataset` **не нарезает батчи внутри конструктора**. Он возвращает **по одному sample** на вызов `__getitem__`. Причина:

- Модели `one_trans` и `rank_mixer` требуют динамического формирования последовательностей разной длины (хотя мы фиксируем `seq_len`, маскирование позволяет работать с реальными длинами).
- Разные модели используют разные наборы признаков, и батчевание через стандартный `DataLoader` с `collate_fn` более гибко.
- Для моделей `catboost` и других, работающих вне PyTorch, удобнее получать плоские массивы без батчевой обёртки.

Таким образом, ожидается, что пользователь создаст `DataLoader` с `batch_size` и, при необходимости, `collate_fn`, которая объединит несколько sample в батч. Для удобства в классе `_ModelSpecificDataset` данные уже представлены в виде тензоров, готовых к батчеванию.

### Пример использования

```python
from torch.utils.data import DataLoader

config = DatasetConfig()
ranking_ds = RankingDataset(config)

# Датасет для OneTrans (train, seq_len=20)
train_ds = ranking_ds.get_dataset('one_trans', split='train', seq_len=20)

# DataLoader с батчем 128
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)

# Пример одного батча
for batch in train_loader:
    print(batch.keys())
    # dict_keys(['dense', 'sparse', 'multivalent', 'label', 'seq_items', 'seq_signals', 'seq_mask'])
    print(batch['seq_items'].shape)   # (128, 20)
    break
```

```python
# Датасет для DCNv2 (только NS-признаки)
train_ds_dcn = ranking_ds.get_dataset('dcn', split='train')
train_loader_dcn = DataLoader(train_ds_dcn, batch_size=2048, shuffle=True)

for batch in train_loader_dcn:
    print(batch.keys())
    # dict_keys(['dense', 'sparse', 'multivalent', 'label'])
    print(batch['dense'].shape)   # (2048, 15)
    break
```

### Особенности мультивалентных признаков

Для каждого sample `multivalent[artist_ids]` – это словарь с полями:
- `values` : тензор `(num_artists_for_this_sample,)` – список ID артистов.
- `lengths` : скалярный тензор (длина списка).

При батчевании через `DataLoader` (стандартная `collate_fn` для словарей) эти структуры не склеиваются автоматически. Рекомендуется использовать кастомную `collate_fn`, которая объединяет `values` в один плоский тензор, а `lengths` – в тензор длин батча (аналогично тому, как это сделано в `RankerDataset` из ДЗ 3). В примере выше мы этого не делаем для простоты, но при необходимости пользователь может реализовать свою `collate_fn`.

### Замечания по эффективности

- Построение истории через `shift` выполняется **один раз** при инициализации `RankingDataset` для заданного `seq_len`. При повторных вызовах `get_dataset` с другим `seq_len` история перестраивается заново (что может быть затратно при частом изменении длины).
- Все тензоры хранятся в оперативной памяти (CPU). Для работы с GPU их нужно переносить в `DataLoader` с помощью кастомной `collate_fn` или в цикле обучения.
- Датасет не применяет `MultihashTransform` (хеширование категориальных признаков) автоматически – это оставлено на усмотрение моделей (например, для HiFormer). При необходимости пользователь может обернуть полученный датасет в дополнительный transform.

### Заключение

`RankingDataset` предоставляет унифицированный интерфейс для загрузки данных для пяти различных архитектур ранжирования. Он скрывает сложность построения временных окон, разделения train/test и преобразования в тензоры, позволяя исследователю сосредоточиться на экспериментировании с моделями.