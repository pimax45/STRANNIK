# CLI трёх программ

## Generator

```bash
python -m generator --help
```

| Флаг | По умолчанию | Назначение |
|---|---:|---|
| `--dem PATH` | обязательный | Исходный NPY, NPZ или GeoTIFF. |
| `--output-dir DIR` | обязательный | Каталог scenario. |
| `--start-x`, `--start-y` | центр | Старт для метрической карты. |
| `--start-lon`, `--start-lat` | центр | Старт для географического GeoTIFF. |
| `--initial-vx`, `--initial-vy` | альтернатива | Начальный вектор скорости: восток/север, м/с. |
| `--heading-deg`, `--initial-speed` | альтернатива | Начальные азимут и скорость вместо компонентов вектора. |
| `--duration SEC` | `60` | Продолжительность. |
| `--frequency-min HZ` | `1` | Минимальная частота измерений. |
| `--frequency-max HZ` | `10` | Максимальная частота измерений. |
| `--constant-frequency` | выключен | Использовать постоянную `frequency-max`. |
| `--sample-rate HZ` | — | Явно задать постоянную частоту 1–10 Гц. |
| `--omit-sample-rate` | выключен | Не записывать известную частоту в контракт локализатора. |
| `--height-type radio\|ground` | `radio` | Значения, записываемые в `heights.txt`. |
| `--omit-baro-altitude` | выключен | Не раскрывать барометрическую высоту локализатору. |
| `--motion MODE` | `straight` | Модель движения. |
| `--baro-altitude M` | `1500` | Барометрическая высота. |
| `--radio-noise-std M` | `0.8` | СКО шума радиовысотомера. |
| `--radio-outlier-probability P` | `0.002` | Вероятность выброса. |
| `--radio-outlier-std M` | `25` | СКО выброса. |
| `--map-radius M` | вся карта | Радиус локальной вырезки; без флага загружается весь GeoTIFF. |
| `--resolution M` | `30` | Шаг подготовленного DEM. |
| `--origin-x`, `--origin-y` | `0` | Начало координат обычного NPY. |
| `--seed N` | `42` | Воспроизводимость. |

Режимы движения: `straight`, `turn`, `yaw`, `speed-change`, `mixed`, `sharp-turn`, `sharp-speed-change`, `sharp-mixed`.

Для GeoTIFF генератор создаёт `heights.txt`, `timestamps.txt` и
`localizer_input.json`. Последний соответствует схеме
`terrain-nav-text-input/v1` и содержит все параметры, необходимые новому CLI
локализатора.

## Localizer

```bash
python -m localizer --help
```

Рекомендуемый контракт:

```bash
python -m localizer --scenario data/run/scenario.json --output-dir results/run
```

| Флаг | Назначение |
|---|---|
| `--scenario PATH` | Manifest генератора. |
| `--input-contract PATH` | Новый `localizer_input.json`; остальные входные флаги читаются из него. |
| `--dem PATH` | Явный метрический DEM вместо manifest. |
| `--measurements PATH` | Явный `radio_samples.npz`. |
| `--heights-text PATH` | Альтернатива NPZ: одна высота в метрах на строку. |
| `--height-type radio\|ground` | Тип строк: радиовысота либо абсолютная высота рельефа. |
| `--initial-state PATH` | Явный `initial_state.json`. |
| `--start-x`, `--start-y` | Начальная точка без JSON. |
| `--initial-vx`, `--initial-vy` | Начальный вектор без JSON. |
| `--heading-deg DEG` | Начальный азимут от севера; используется с `--heights-text`. |
| `--initial-speed MPS` | Начальная путевая скорость; используется с `--heights-text`. |
| `--sample-rate HZ` | Известная постоянная частота 1–10 Гц; без флага она оценивается. |
| `--baro-altitude M` | Абсолютная высота БПЛА для преобразования радиовысот. |
| `--frequency-min`, `--frequency-max` | Диапазон оценки частоты, по умолчанию 1–10 Гц. |
| `--frequency-step HZ` | Шаг оценки частоты, по умолчанию `0.25`. |
| `--frequency-prefix-samples N` | Число начальных отсчётов для оценки частоты, по умолчанию `31`. |
| `--map-radius M` | Радиус рабочей вырезки; без флага загружается весь GeoTIFF. |
| `--working-resolution M` | Разрешение метрической рабочей DEM; по умолчанию берётся из config. |
| `--config PATH` | Необязательная YAML-конфигурация поиска. |
| `--truth PATH` | Истинный путь только для графика. |
| `--output-dir DIR` | Обязательный каталог результатов. |
| `--preview-trajectory` | Открыть график перед сохранением. |
| `--no-plots` | Сохранить только JSON. |

В YAML поиска можно задать `absolute_height_sigma_m` — ожидаемую суммарную вертикальную ошибку в метрах. Значение по умолчанию `10.0`; чем оно меньше, тем сильнее абсолютное несовпадение высот понижает гипотезу. Итоговый `result.json` содержит компонент `scores.absolute_height_likelihood`.

Усиленный фильтр выбросов задаётся параметрами `hampel_window_size: 7`, `hampel_n_sigma: 3.0`, `hampel_passes: 2`. Уменьшение `hampel_n_sigma` или увеличение числа проходов делает фильтрацию агрессивнее.

Динамические ограничения по умолчанию: `window_seconds: 10`, `local_turn_limit_deg: 12`, `max_delta_v_mps_per_window: 4`, глобальная скорость `0.6…1.4` от начальной. Порог неоднозначности задаётся `confidence_min_score_gap: 0.01`.

Пример нового текстового контракта:

```bash
python -m localizer \
  --dem map.tif \
  --heights-text heights.txt \
  --height-type radio \
  --start-x 65.2 --start-y 67.2 \
  --heading-deg 73 --initial-speed 50 \
  --sample-rate 3 --baro-altitude 1500 \
  --output-dir results/text
```

Если файл уже сформирован generator, рекомендуемый запуск короче:

```bash
python -m localizer \
  --input-contract data/demo/localizer_input.json \
  --output-dir results/demo
```

Команда `make localize` использует `LOCALIZER_INPUT`, по умолчанию
`data/demo/localizer_input.json`, и не берёт входные навигационные параметры из
Makefile.

Для автоматической оценки частоты удалите `--sample-rate`. Если отсутствует и
`--baro-altitude`, первый радиовысотный отсчёт привязывается к DEM в стартовой
точке; это соответствует предположению о постоянной абсолютной высоте полёта.

## Orchestrator

```bash
python -m orchestrator --help
```

Принимает параметры generator, но использует `--dataset-dir` для его файлов и
`--output-dir` для результата localizer. Дополнительно доступны
`--frequency-step`, `--search-config`, `--preview-trajectory`, `--no-plots`.

Orchestrator выполняет именно два subprocess-вызова:

```text
python -m generator ...
python -m localizer --dem ... --heights-text <dataset-dir>/heights.txt ...
```

Оркестратор получает параметры второго вызова из
`<dataset-dir>/localizer_input.json`. При `--omit-sample-rate` частота во второй
вызов не передаётся и определяется локализатором.

## Batch runner: 128 тестов

```bash
python -m orchestrator.batch \
  --dem map.tif \
  --output-dir results_test \
  --start-x 502000 --start-y 6998000 \
  --heading-deg 73
```

Команда по умолчанию запускает скорости `20 30 40 50`, длительности
`100 200 300 400`, частоты `1 3 7 10` и режимы `straight turn`. Эти списки можно
переопределить флагами `--speeds`, `--durations`, `--frequencies`, `--motions`,
например для короткой проверки:

```bash
python -m orchestrator.batch \
  --dem map.tif --output-dir results_test_smoke \
  --start-x 502000 --start-y 6998000 \
  --speeds 20 --durations 100 --frequencies 3 --motions straight
```

| Флаг | Назначение |
|---|---|
| `--output-dir` | Корневой каталог; по умолчанию `results_test`. |
| `--timeout SEC` | Максимальное время одного теста. |
| `--force` | Не пропускать уже завершённые тесты. |
| `--fail-fast` | Остановиться после первой ошибки. |
| `--omit-sample-rate` | Проверять оценку частоты вместо передачи точного значения. |
| `--search-config` | YAML-конфигурация localizer для всех тестов. |

Изображения собираются в `results_test/image` с именами вида
`trajectory_20_100_1_straight.png`. Общие результаты записываются в
`summary.csv` и `summary.json`; подробные файлы каждого запуска находятся в
`runs/<test-id>`.
