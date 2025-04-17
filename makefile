# Makefile
.PHONY: build run clean help

# Variables
PROJECT_NAME := ae-model

# Build the Docker image
build:
	docker-compose build

# Run the container (build first if needed)
run: build
	docker-compose run --rm $(PROJECT_NAME)

# Build and run in one command
all: build run

# Clean Docker artifacts
clean:
	docker-compose down --rmi all --volumes
	docker system prune -f

# Help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build     Build the Docker image"
	@echo "  run       Run the container (builds first if needed)"
	@echo "  all       Build and run in one step"
	@echo "  clean     Remove Docker containers, images, and volumes"
	@echo "  help      Show this help message"