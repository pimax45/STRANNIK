# Terrain Nav

Проект состоит ровно из трёх независимых программ:

```text

localizer/      DEM + измерения + начальное состояние -> оценка навигации

```

## Установка

Требуется Python 3.13+.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

После установки доступны три команды:

```text
terrain-nav-localizer
```

Без установки editable-пакета используйте `python -m localizer`.

## 1. Generator

Генератор принимает внешний NPY/NPZ/GeoTIFF, строит непрерывную билинейную функцию высоты, моделирует маршрут и формирует измерения радиовысотомера. Для GeoTIFF он дополнительно создаёт новый текстовый контракт: `heights.txt` с одним значением на строку и `localizer_input.json` с исходной картой, стартом, направлением, скоростью и доступными параметрами измерений.

Пример для текущего файла `output_hh_1.tif`:

```bash
.venv/bin/python -m generator \
  --dem output_hh_1.tif \
  --output-dir data/cop30 \
  --start-lon 65.2 \
  --start-lat 67.2 \
  --initial-vx 47.8 \
  --initial-vy 14.6 \
  --duration 60 \
  --frequency-min 1 \
  --frequency-max 10 \
  --baro-altitude 1600 \
  --radio-noise-std 0.8 \
  --map-radius 10000
```

Географический GeoTIFF переводится в локальную UTM-зону. Если передан
`--map-radius`, загружается квадратная вырезка вокруг старта; без этого флага
используется вся карта. Выход:

```text
data/cop30/
├── scenario.json
├── localizer_input.json
├── heights.txt
├── timestamps.txt
├── dem.npz
├── radio_samples.npz
├── initial_state.json
└── true_path.npz
```

`radio_samples.npz` содержит `timestamps`, `radio_altitude_m`, `baro_altitude_m`. `initial_state.json` содержит `x`, `y`, `velocity_x_mps`, `velocity_y_mps`.

Новый рекомендуемый вызов использует непосредственно направление и скорость:

```bash
.venv/bin/python -m generator \
  --dem output_hh_1.tif \
  --output-dir data/text-demo \
  --start-lon 65.2 --start-lat 67.2 \
  --heading-deg 73 --initial-speed 50 \
  --duration 60 --sample-rate 3 \
  --baro-altitude 1600
```

`--sample-rate` создаёт постоянную частоту. Без этого флага сохраняется прежний
режим нерегулярных измерений в диапазоне `--frequency-min..--frequency-max`.
Флаг `--omit-sample-rate` скрывает даже известную постоянную частоту в
`localizer_input.json`, чтобы проверить её восстановление локализатором.

## 2. Localizer

Локализатор ничего не знает о способе генерации данных. Он читает метрический DEM, измерения и начальный вектор скорости, после чего выполняет beam search по азимуту и физической путевой скорости.

Score учитывает как форму, так и абсолютную высоту. Для абсолютных значений вычисляется Gaussian likelihood-like компонент:

```text
absolute_height_likelihood = exp(-0.5 * (absolute_RMSE / sigma_height)^2)
```

По умолчанию `sigma_height = 10 м` и компонент имеет вес 20% на всех этапах: векторизованный отбор, re-ranking окна и переоценка полной истории. Параметр `absolute_height_sigma_m` должен соответствовать суммарной ошибке DEM, барометра и радиовысотомера; при большом неизвестном вертикальном bias его следует увеличить.

Перед сопоставлением измеренный профиль проходит усиленный фильтр Хампеля: два прохода, локальное окно 7 измерений и порог 3σ по median absolute deviation. После него применяется скользящее среднее с окном 3. Параметры `hampel_window_size`, `hampel_n_sigma`, `hampel_passes` настраиваются в YAML поиска.

Поиск использует полные 10-секундные окна с 50% перекрытием. Переходная модель ограничивает локальный поворот до 12°, изменение скорости до 4 м/с за шаг и удерживает скорость в диапазоне `0.6…1.4` от начальной. Если разрыв двух лучших гипотез меньше `0.01`, confidence принудительно считается низким.

```bash
.venv/bin/python -m localizer \
  --scenario data/cop30/scenario.json \
  --output-dir results/cop30
```

Входы можно передать явно вместо manifest:

```bash
.venv/bin/python -m localizer \
  --dem data/cop30/dem.npz \
  --measurements data/cop30/radio_samples.npz \
  --initial-state data/cop30/initial_state.json \
  --output-dir results/cop30
```

Выход: `result.json`, `trajectory.png`, `score_heatmap.png`, `profile_comparison.png`.

### GeoTIFF + текстовый файл высот

Локализатор также работает напрямую с исходным GeoTIFF и текстовым файлом, в
котором каждая непустая строка содержит одно значение высоты в метрах. Для
географического GeoTIFF `--start-x` и `--start-y` означают долготу и широту; для
проекционного GeoTIFF это координаты в CRS самой карты.

Если параметры собраны в созданном генератором `localizer_input.json`, их не
нужно повторять в командной строке:

```bash
.venv/bin/python -m localizer \
  --input-contract data/demo/localizer_input.json \
  --output-dir results/demo
```

То же через Makefile:

```bash
make localize
```

Перед сохранением `trajectory.png` откроется интерактивное окно Matplotlib.
PNG записывается после закрытия окна. Для запуска без графического интерфейса:

```bash
make localize PREVIEW=0
```

Другой контракт можно указать так:

```bash
make localize LOCALIZER_INPUT=data/other/localizer_input.json
```

Если строки содержат показания радиовысотомера (расстояние до земли), используйте
режим по умолчанию `--height-type radio`. Барометрическую высоту можно передать
через `--baro-altitude`. Если её нет, программа привязывает первый отсчёт к
высоте DEM в стартовой точке и предполагает постоянную абсолютную высоту полёта.
Если файл уже содержит абсолютные высоты рельефа, задайте `--height-type ground`.

С известной постоянной частотой:

```bash
.venv/bin/python -m localizer \
  --dem map.tif \
  --heights-text radio_heights.txt \
  --height-type radio \
  --start-x 65.2 \
  --start-y 67.2 \
  --heading-deg 73 \
  --initial-speed 50 \
  --sample-rate 3 \
  --baro-altitude 1500 \
  --output-dir results/text-input
```

Если `--sample-rate` не указан, программа перебирает эффективную постоянную
частоту в диапазоне 1–10 Гц и выбирает временную шкалу, на которой начальный
профиль лучше совпадает с DEM вдоль известных начальных направления и скорости:

```bash
.venv/bin/python -m localizer \
  --dem map.tif \
  --heights-text ground_heights.txt \
  --height-type ground \
  --start-x 65.2 \
  --start-y 67.2 \
  --heading-deg 73 \
  --initial-speed 50 \
  --output-dir results/text-input-inferred-rate
```

В этом режиме дополнительно создаются:

```text
trajectory_coordinates.csv   локальные, проекционные и WGS84-координаты
trajectory_coordinates.json  тот же набор точек с описанием систем координат
trajectory.png               визуализация оценённой траектории на DEM
```

Текстовый файл без временных меток позволяет восстановить только одну
эффективную постоянную частоту. Произвольно меняющиеся интервалы между
измерениями однозначно определить по одним высотам нельзя; для таких данных
следует использовать существующий NPZ-контракт с точными `timestamps`.

`map_radius_m` не является обязательным входным параметром. Его отсутствие или
значение `null` в `localizer_input.json` означает загрузку всего GeoTIFF. Явный
радиус полезен только для снижения расхода памяти на больших картах.

## 3. Orchestrator

Оркестратор принимает параметры генератора и запускает его отдельным процессом.
Затем он читает созданный `localizer_input.json` и отдельным процессом запускает
локализатор с GeoTIFF, `heights.txt`, стартом, азимутом и скоростью. Legacy
`scenario.json` в этой цепочке больше не используется.

```bash
.venv/bin/python -m orchestrator \
  --dem output_hh_1.tif \
  --dataset-dir data/cop30 \
  --output-dir results/cop30 \
  --start-lon 65.2 \
  --start-lat 67.2 \
  --heading-deg 73 \
  --initial-speed 50 \
  --duration 60 \
  --sample-rate 3 \
  --baro-altitude 1600
```

Для проверки режима с неизвестной частотой добавьте `--omit-sample-rate`. Для
нерегулярной частоты не задавайте `--sample-rate`: в текстовом контракте точные
timestamps сохраняются только как эталонный `timestamps.txt`, а локализатор
оценивает одну эффективную постоянную частоту.

## 4. Пакет из 128 тестов

Пакетный runner перебирает все сочетания:

```text
скорость:    20, 30, 40, 50 м/с
duration:    100, 200, 300, 400 с
частота:     1, 3, 7, 10 Гц
motion:      straight, turn
```

Итого: `4 × 4 × 4 × 2 = 128` запусков generator → localizer.

```bash
.venv/bin/python -m orchestrator.batch \
  --dem output_hh_1.tif \
  --output-dir results_test \
  --start-lon 59.5 --start-lat 67.2 \
  --heading-deg 73 \
  --baro-altitude 1500
```

Или через Makefile:

```bash
make batch-tests
```

Результаты имеют следующую структуру:

```text
results_test/
├── image/
│   ├── trajectory_20_100_1_straight.png
│   ├── trajectory_20_100_1_turn.png
│   └── ...
├── runs/<test-id>/
│   ├── result.json
│   ├── trajectory.png
│   ├── trajectory_coordinates.csv
│   ├── batch_metrics.json
│   └── run.log
├── datasets/<test-id>/
│   ├── heights.txt
│   ├── localizer_input.json
│   └── true_path.npz
├── summary.csv
└── summary.json
```

`summary.csv` содержит время выполнения, confidence, score, ошибку координат,
азимута и скорости. Завершённые тесты при повторном запуске пропускаются; флаг
`--force` запускает их заново. Ошибка одного сценария записывается в summary и
не останавливает оставшиеся тесты. Полная матрица вычислительно тяжёлая и может
выполняться продолжительное время.

Полный перечень флагов приведён в [PIPELINE_CLI.md](PIPELINE_CLI.md). Быстрые команды доступны через `make generator`, `make localize`, `make orchestrator`, `make batch-tests`; `make localizer` оставлен как алиас.

## Ограничения

- алгоритм плохо локализуется на плоском и повторяющемся рельефе;
- результат зависит от вертикальной точности DEM, барометра и радиовысотомера;
- GeoTIFF для рабочей стадии переводится генератором в метрическую сетку;
- оценка частоты без timestamp опирается на известные начальные скорость и
  направление и может быть неоднозначна на плоском или повторяющемся рельефе;
- это навигационная оценка, а не автопилот или система управления.
