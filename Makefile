PYTHON ?= .venv/bin/python

DATA_DIR ?= data/demo
RESULTS_DIR ?= results/demo

TERRAIN ?= mixed
MOTION ?= sharp-turn
BARO_ALTITUDE ?= 1500
SAMPLE_RATE ?= 3
DURATION ?= 120
START_X ?= 1000
START_Y ?= 4300
INITIAL_SPEED ?= 50
AZIMUTH ?= 80
RESOLUTION ?= 30
RADIO_NOISE_STD ?= 2.5
RADIO_OUTLIER_PROBABILITY ?= 0.03
SEED ?= 42

DEM ?= $(DATA_DIR)/dem.npy
CONFIG ?= $(DATA_DIR)/config.yaml
NMEA ?= $(DATA_DIR)/radio_altimeter.nmea
INITIAL_AZIMUTH ?= $(AZIMUTH)
PREVIEW ?= 0
VERIFY_CHECKSUM ?= 0

INITIAL_AZIMUTH_ARG = $(if $(strip $(INITIAL_AZIMUTH)),--initial-azimuth $(INITIAL_AZIMUTH),)
PREVIEW_ARG = $(if $(filter 1 true yes,$(PREVIEW)),--preview-trajectory,)
CHECKSUM_ARG = $(if $(filter 1 true yes,$(VERIFY_CHECKSUM)),--verify-checksum,)

.PHONY: help install generate-test localize localize-preview demo test

help:
	@echo "Доступные цели:"
	@echo "  make generate-test       создать синтетический набор"
	@echo "  make localize            выполнить локализацию"
	@echo "  make localize-preview    локализация с предпросмотром trajectory.png"
	@echo "  make demo                generate-test, затем localize"
	@echo "  make test                запустить тесты"
	@echo "  make install             установить requirements.txt"
	@echo "Параметры можно переопределять: make generate-test MOTION=sharp-mixed START_X=1500 START_Y=1200"

install:
	$(PYTHON) -m pip install -r requirements.txt

generate-test:
	$(PYTHON) -m terrain_nav generate-test \
		--terrain $(TERRAIN) \
		--motion $(MOTION) \
		--output-dir $(DATA_DIR) \
		--baro-altitude $(BARO_ALTITUDE) \
		--sample-rate $(SAMPLE_RATE) \
		--duration $(DURATION) \
		--start-x $(START_X) \
		--start-y $(START_Y) \
		--initial-speed $(INITIAL_SPEED) \
		--azimuth $(AZIMUTH) \
		--resolution $(RESOLUTION) \
		--radio-noise-std $(RADIO_NOISE_STD) \
		--radio-outlier-probability $(RADIO_OUTLIER_PROBABILITY) \
		--seed $(SEED)

localize:
	$(PYTHON) -m terrain_nav localize \
		--dem $(DEM) \
		--config $(CONFIG) \
		--nmea $(NMEA) \
		--start-x $(START_X) \
		--start-y $(START_Y) \
		--initial-speed $(INITIAL_SPEED) \
		--baro-altitude $(BARO_ALTITUDE) \
		--output-dir $(RESULTS_DIR) $(INITIAL_AZIMUTH_ARG) $(CHECKSUM_ARG) $(PREVIEW_ARG) \
		--preview-trajectory

localize-preview:
	$(MAKE) localize PREVIEW=1

demo: generate-test localize

test:
	$(PYTHON) -m pytest
