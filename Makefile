PYTHON ?= .venv/bin/python

SOURCE_DEM ?= output_hh_1.tif
DATASET_DIR ?= data/demo
RESULTS_DIR ?= results/demo
BATCH_RESULTS_DIR ?= results_test
LOCALIZER_INPUT ?= $(DATASET_DIR)/localizer_input.json
SEARCH_CONFIG ?=

# Для географического GeoTIFF задайте START_LON/START_LAT. Для проекционного
# GeoTIFF очистите их и задайте START_X/START_Y.
START_X ?=
START_Y ?=
START_LON ?= 59.5
START_LAT ?= 67.2

HEADING_DEG ?= 73
INITIAL_SPEED ?= 30
DURATION ?= 300
MOTION ?= turn

# SAMPLE_RATE=3 означает известную постоянную частоту. SAMPLE_RATE= включает
# нерегулярную генерацию FREQUENCY_MIN..FREQUENCY_MAX и оценку частоты.
SAMPLE_RATE ?= 3
FREQUENCY_MIN ?= 1
FREQUENCY_MAX ?= 10
FREQUENCY_STEP ?= 0.25
CONSTANT_FREQUENCY ?= 0
OMIT_SAMPLE_RATE ?= 0

HEIGHT_TYPE ?= radio
BARO_ALTITUDE ?= 1500
OMIT_BARO_ALTITUDE ?= 0
RADIO_NOISE_STD ?= 0.8
RADIO_OUTLIER_PROBABILITY ?= 0.002
RADIO_OUTLIER_STD ?= 25
# Пустое значение означает загрузку всей карты.
MAP_RADIUS ?=
RESOLUTION ?= 30
ORIGIN_X ?= 0
ORIGIN_Y ?= 0
SEED ?= 42
PREVIEW ?= 0
NO_PLOTS ?= 0

GENERATOR_START_ARG = $(if $(strip $(START_LON)),--start-lon $(START_LON) --start-lat $(START_LAT),--start-x $(START_X) --start-y $(START_Y))
MOTION_STATE_ARG = --heading-deg $(HEADING_DEG) --initial-speed $(INITIAL_SPEED)

GENERATOR_RATE_ARG = $(if $(strip $(SAMPLE_RATE)),--sample-rate $(SAMPLE_RATE),--frequency-min $(FREQUENCY_MIN) --frequency-max $(FREQUENCY_MAX) $(if $(filter 1 true yes,$(CONSTANT_FREQUENCY)),--constant-frequency,))
OMIT_RATE_ARG = $(if $(filter 1 true yes,$(OMIT_SAMPLE_RATE)),--omit-sample-rate,)
OMIT_BARO_ARG = $(if $(filter 1 true yes,$(OMIT_BARO_ALTITUDE)),--omit-baro-altitude,)

CONFIG_ARG = $(if $(strip $(SEARCH_CONFIG)),--config $(SEARCH_CONFIG),)
ORCHESTRATOR_CONFIG_ARG = $(if $(strip $(SEARCH_CONFIG)),--search-config $(SEARCH_CONFIG),)
PREVIEW_ARG = $(if $(filter 1 true yes,$(PREVIEW)),--preview-trajectory,)
NO_PLOTS_ARG = $(if $(filter 1 true yes,$(NO_PLOTS)),--no-plots,)
MAP_RADIUS_ARG = $(if $(strip $(MAP_RADIUS)),--map-radius $(MAP_RADIUS),)

.PHONY: help install generator localize localizer orchestrator batch-tests test

help:
	@echo "Новый протокол GeoTIFF + heights.txt:"
	@echo "  make generator      создать heights.txt и localizer_input.json"
	@echo "  make localize       запустить localizer по localizer_input.json"
	@echo "  make localizer      алиас для make localize"
	@echo "  make orchestrator   последовательно выполнить обе стадии"
	@echo "  make batch-tests    запустить матрицу из 128 тестов"
	@echo "  make test           запустить тесты"
	@echo ""
	@echo "Известная частота: SAMPLE_RATE=3"
	@echo "Скрытая частота:   SAMPLE_RATE=3 OMIT_SAMPLE_RATE=1"
	@echo "Нерегулярная:      SAMPLE_RATE="
	@echo "Вся карта:         MAP_RADIUS="

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

generator:
	$(PYTHON) -m generator \
		--dem $(SOURCE_DEM) \
		--output-dir $(DATASET_DIR) \
		$(GENERATOR_START_ARG) \
		$(MOTION_STATE_ARG) \
		--duration $(DURATION) \
		$(GENERATOR_RATE_ARG) $(OMIT_RATE_ARG) \
		--height-type $(HEIGHT_TYPE) \
		--motion $(MOTION) \
		--baro-altitude $(BARO_ALTITUDE) $(OMIT_BARO_ARG) \
		--radio-noise-std $(RADIO_NOISE_STD) \
		--radio-outlier-probability $(RADIO_OUTLIER_PROBABILITY) \
		--radio-outlier-std $(RADIO_OUTLIER_STD) \
		$(MAP_RADIUS_ARG) \
		--resolution $(RESOLUTION) \
		--origin-x $(ORIGIN_X) \
		--origin-y $(ORIGIN_Y) \
		--seed $(SEED)

# Интерактивное окно показывается до сохранения trajectory.png.
# Значение из командной строки (PREVIEW=0) имеет приоритет.
localize: PREVIEW = 1
localize:
	$(PYTHON) -m localizer \
		--input-contract $(LOCALIZER_INPUT) \
		--frequency-step $(FREQUENCY_STEP) \
		--output-dir $(RESULTS_DIR) \
		$(CONFIG_ARG) $(PREVIEW_ARG) $(NO_PLOTS_ARG)

localizer: localize

orchestrator:
	$(PYTHON) -m orchestrator \
		--dem $(SOURCE_DEM) \
		--dataset-dir $(DATASET_DIR) \
		--output-dir $(RESULTS_DIR) \
		$(GENERATOR_START_ARG) \
		$(MOTION_STATE_ARG) \
		--duration $(DURATION) \
		$(GENERATOR_RATE_ARG) $(OMIT_RATE_ARG) \
		--frequency-step $(FREQUENCY_STEP) \
		--height-type $(HEIGHT_TYPE) \
		--motion $(MOTION) \
		--baro-altitude $(BARO_ALTITUDE) $(OMIT_BARO_ARG) \
		--radio-noise-std $(RADIO_NOISE_STD) \
		--radio-outlier-probability $(RADIO_OUTLIER_PROBABILITY) \
		--radio-outlier-std $(RADIO_OUTLIER_STD) \
		$(MAP_RADIUS_ARG) \
		--resolution $(RESOLUTION) \
		--origin-x $(ORIGIN_X) \
		--origin-y $(ORIGIN_Y) \
		--seed $(SEED) \
		$(ORCHESTRATOR_CONFIG_ARG) $(PREVIEW_ARG) $(NO_PLOTS_ARG)

batch-tests:
	$(PYTHON) -m orchestrator.batch \
		--dem $(SOURCE_DEM) \
		--output-dir $(BATCH_RESULTS_DIR) \
		$(GENERATOR_START_ARG) \
		--heading-deg $(HEADING_DEG) \
		--height-type $(HEIGHT_TYPE) \
		--baro-altitude $(BARO_ALTITUDE) $(OMIT_BARO_ARG) \
		$(OMIT_RATE_ARG) \
		--frequency-step $(FREQUENCY_STEP) \
		--radio-noise-std $(RADIO_NOISE_STD) \
		--radio-outlier-probability $(RADIO_OUTLIER_PROBABILITY) \
		--radio-outlier-std $(RADIO_OUTLIER_STD) \
		$(MAP_RADIUS_ARG) \
		--resolution $(RESOLUTION) \
		--seed $(SEED) \
		$(ORCHESTRATOR_CONFIG_ARG)

test:
	PYTHONPATH=. $(PYTHON) -m pytest
