.PHONY: build map preprocess clean help

# Variables
PROJECT_NAME := ae-model

# Build the Docker image
build:
	docker-compose build

# Step 1: Dataset Mapping (mapped_dataset.csv)
map: build
	docker-compose run --rm $(PROJECT_NAME) python src/map.py
	@echo "Step 1 (Mapping) completed. Check ./results/mapped_dataset.csv"

# Step 2: Preprocessing (mapped_dataset.csv)
preprocess: build
	docker-compose run --rm $(PROJECT_NAME) python src/preprocess.py
	@echo "Step 3 (Preprocessing) completed. Check ./results/processed_data/"

# Step 3: Feature extraction
features: build
	docker-compose run --rm $(PROJECT_NAME) python src/featextract.py
	@echo "Feature extraction completed. Check ./results/features/"

# Clean Docker artifacts
clean:
	docker-compose down --rmi all --volumes
	docker system prune -f

# Help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       Build the Docker image"
	@echo "  map         Run Step 1 (map WAV files to CSV metadata)"
	@echo "  preprocess  Run Step 2 (preprocess signals and combine channels)"
	@echo "  features    Run Step 3 (feature extraction based on preprocessed data)"
	@echo "  clean       Remove Docker containers, images, and volumes"
	@echo "  help        Show this help message"