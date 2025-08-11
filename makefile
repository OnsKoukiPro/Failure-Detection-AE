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

# Step 1 FOR KI: Dataset Mapping (metadata.csv)
map-ki: build
	docker-compose run --rm $(PROJECT_NAME) python src/KI/map.py
	@echo "Step 1 (Mapping) completed. Check ./results/KI/metadata.csv"

# Step 2: Preprocessing (mapped_dataset.csv)
preprocess: build
	docker-compose run --rm $(PROJECT_NAME) python src/preprocess.py
	@echo "Step 2  (Preprocessing) completed. Check ./results/processed_data/"

# Step 2 bis : Preprocessing / augmenting data (augmentdata.csv)
augment: build
	docker-compose run --rm $(PROJECT_NAME) python src/augmentdata.py
	@echo "Augmentation completed. Check ./results/augmented_data/"

# Step 2 FOR KI: Preprocessing (metadata.csv)
process-ki: build
	docker-compose run --rm $(PROJECT_NAME) python src/KI/preprocess.py
	@echo "Step 2 (Preprocessing) completed. Check ./results/KI/processed"

# Step 3: Feature extraction
features: build
	docker-compose run --rm $(PROJECT_NAME) python src/featextract.py
	@echo "Feature extraction completed. Check ./results/features/"

# Step 3: Feature extraction unsupervised mode, no augmented data
features-: build
	docker-compose run --rm $(PROJECT_NAME) python src/featextractunsup.py
	@echo "Feature extraction completed. Check ./results/features/"

# Step 3: Feature extraction unsupervised mode, no augmented data ,KI
features-ki: build
	docker-compose run --rm $(PROJECT_NAME) python src/KI/featextract.py
	@echo "Feature extraction completed. Check ./results/KI/features"

# Step 4: Data Split (Test / Train)
split: build
	docker-compose run --rm $(PROJECT_NAME) python src/splitdata.py
	@echo "Feature extraction completed. Check ./results/splits/"

# Step 5: Train Model
train: build
	docker-compose run --rm $(PROJECT_NAME) python src/train.py \
		--train_csv /app/results/splits/stratified_train.csv \
		--test_csv /app/results/splits/stratified_test.csv \

# Step 5: Train Unsupervised Model Isolation Forest (Anomaly Detection)
# DBSCAN (Clustering)
train-: build
	docker-compose run --rm ae-model python src/trainunsup.py

# Step 5: Train Unsupervised Model KI
train-ki: build
	docker-compose run --rm ae-model python src/KI/train.py

# Step 6: Test Model
test: train
	docker-compose run --rm $(PROJECT_NAME) python src/test.py

# Step 7: Validate Model
validate:
	docker-compose run --rm $(PROJECT_NAME) python src/validation.py --csv_path /app/results/comparison/predictions_comparison.csv --output_dir /app/results/analysis

# Step 7: Validate Unsupervised Model
validate-:
	docker-compose run --rm ae-model python src/validateunsup.py

# Step 7: Validate Unsupervised Model KI
validate-ki:
	docker-compose run --rm ae-model python src/KI/analyse.py

realtime:
	docker-compose run --rm $(PROJECT_NAME) python src/realtime_test.py

graph:
	docker-compose run --rm ae-model python src/graph.py

analyse:
	docker-compose run --rm ae-model python src/analyse.py
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
	@echo "  split       Run Step 4 (create train/val/test splits)"
	@echo "  clean       Remove Docker containers, images, and volumes"
	@echo "  help        Show this help message"