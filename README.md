# RLHF-UI

A user interface to collect preference data, train reward models, and perform Reinforcement Learning from Human Feedback (RLHF).

![Capture d’écran 2025-04-07 à 15 14 22](https://github.com/user-attachments/assets/f83c16c1-1744-46f4-9996-0546a63ba583)


## Description

This project provides a comprehensive graphical user interface for the RLHF pipeline, including:

1. **Preference Data Collection**: Collect human feedback on pairs of images
2. **Reward Model Training**: Train models to predict human preferences
3. **Model Inference**: Use trained models to evaluate new content
4. **RLHF Optimization**: Fine-tune generative models using the reward model

## Features

- Interactive preference collection with multiple sampling strategies
- Reward model training with real-time visualization via Weights & Biases
- Integrated model inference on new data
- RLHF optimization interface
- Clean, user-friendly Gradio UI

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RLHF-UI.git
cd RLHF-UI

# Install dependencies using Poetry
poetry install

# Alternatively, if you don't use Poetry:
pip install -e .
```

## Usage

You can run the application in several ways:

```bash
# Using Poetry
poetry run rlhf-ui

# As a Python module
poetry run python -m rlhf_ui

# Or without Poetry
python -m rlhf_ui
```

The application interface has four main tabs:

1. **Collect Preferences**: Upload image pairs and collect preference data
2. **Train Reward Model**: Train a reward model using the collected preferences
3. **Reward Model Inference**: Use the trained model to evaluate new images
4. **RLHF Optimization**: Fine-tune generative models with the reward model

### Configuration

The application uses a default configuration, but you can customize it by:

1. Creating a `.env` file in the project root
2. Setting environment variables like `IMAGE_FOLDER`, `OUTPUT_FOLDER`, etc.

### Visualization with Weights & Biases

This project integrates Weights & Biases for real-time training visualization and experiment tracking:

1. Sign up for a free account at [wandb.ai](https://wandb.ai) if you don't have one
2. Login via command line: `wandb login`
3. When training reward models, a link to the W&B dashboard will appear in the UI

## Development

### Project Structure

```
RLHF-UI/
├── src/
│   └── rlhf_ui/
│       ├── app.py             # Main Gradio application
│       ├── config.py          # Configuration management
│       ├── __main__.py        # Module entry point
│       ├── data/              # Data handling modules
│       │   ├── dataset.py     # Dataset definitions
│       │   ├── sampler.py     # Image pair sampling
│       │   └── storage.py     # Preference data storage
│       ├── models/            # Model definitions and training
│       │   ├── embedding.py   # Image embedding models
│       │   └── trainer.py     # Reward model training
│       ├── rlhf/              # RLHF optimization
│       │   └── optimizer.py   # RLHF optimizer implementation
│       ├── ui/                # UI components (structure for future refactoring)
│       │   └── tabs/          # Individual tab implementations
│       │       └── base.py    # Base tab class
│       └── visualization/     # Visualization with Weights & Biases
│           └── logger.py      # W&B integration
├── preference_data/           # Collected preference data
├── reward_model/              # Trained model checkpoints
├── pyproject.toml             # Project dependencies
└── README.md
```

### Code Architecture

The codebase is organized around the following components:

1. **Data Handling** (`rlhf_ui.data`): Classes for storing, sampling, and preparing preference data
2. **Models** (`rlhf_ui.models`): Reward model definition and training logic
3. **RLHF** (`rlhf_ui.rlhf`): RLHF optimization and fine-tuning
4. **Visualization** (`rlhf_ui.visualization`): Integration with Weights & Biases
5. **UI** (`rlhf_ui.ui`): Gradio UI components and tabs

### Refactoring Plans

The main `app.py` file (1400+ lines) should eventually be refactored into smaller components:

1. **UI Tabs**: Move each tab into its own class in the `rlhf_ui.ui.tabs` package
2. **Event Handlers**: Organize event handlers by functionality
3. **Main App**: Keep the main app class focused on high-level orchestration

A base tab class (`BaseTab` in `ui/tabs/base.py`) has been created as a foundation for this refactoring. Each tab should:

- Inherit from `BaseTab`
- Implement the `build` and `register_event_handlers` methods
- Be registered with the main app

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines

1. **Code Style**: Follow [PEP 8](https://peps.python.org/pep-0008/) and use type hints
2. **Documentation**: Document classes and functions with docstrings
3. **Testing**: Add tests for new functionality
4. **Dependencies**: Use Poetry for dependency management

## Future Work

- Add a tab for inference on a given dataset of images. Should allow to use a model, run it on the images and get scores. Maybe even allow to sort/filter the images by score and export the filtered subsets.
- Implement a more flexible model saving strategy - save only the best checkpoint or all checkpoints based on user preference
- Support for different model architectures and training techniques
- Complete the UI refactoring to use the tab class structure
- Add a proper test suite
