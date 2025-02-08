SHELL := /bin/bash

BASE.DIR=$(PWD)
DATA.DIR=$(BASE.DIR)/data
DOWNLOADS.DIR=$(BASE.DIR)/downloads

MODEL.URL=https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip&dl=1
models.fetch: .FORCE
	mkdir -p $(DOWNLOADS.DIR)
	cd $(DOWNLOADS.DIR) && wget $(MODEL.URL)

MODEL.DIR=$(BASE.DIR)/models
models.install: .FORCE
	unzip -d $(MODEL.DIR) -q $(DOWNLOADS.DIR)/models.zip


.FORCE: