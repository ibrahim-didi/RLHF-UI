# src/rlhf_ui/cli.py
"""
Command-line interface for the RLHF UI application.
"""

import sys
import logging
import random
from pathlib import Path

import click

from rlhf_ui.config import load_config, save_config
from rlhf_ui.main import main as start_app

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--image-folder', type=click.Path(exists=True, file_okay=False),
              help='Path to folder containing images')
@click.option('--output-folder', type=click.Path(file_okay=False),
              help='Path to save preference data')
@click.option('--model-output-dir', type=click.Path(file_okay=False),
              help='Directory to save model checkpoints')
@click.option('--config', type=click.Path(exists=True, dir_okay=False),
              help='Path to configuration file')
@click.option('--sampling-strategy', type=click.Choice(['random', 'active', 'diversity']),
              help='Strategy for sampling image pairs')
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
def cli(ctx, image_folder, output_folder, model_output_dir, config, sampling_strategy, debug):
    """
    RLHF UI - Tool for collecting human preferences and training reward models.
    
    Run without commands to start the GUI application.
    """
    # Configure logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    if config:
        app_config = load_config(Path(config))
    else:
        app_config = load_config()
    
    # Override with command line options
    if image_folder:
        app_config.image_folder = Path(image_folder)
    if output_folder:
        app_config.output_folder = Path(output_folder)
    if model_output_dir:
        app_config.model_output_dir = Path(model_output_dir)
    if sampling_strategy:
        app_config.sampling_strategy = sampling_strategy
    
    # Store config in context for subcommands
    ctx.obj = {
        'config': app_config
    }
    
    # If no subcommand is invoked, start the application
    if ctx.invoked_subcommand is None:
        return start_app(app_config)


@cli.command()
@click.pass_context
@click.option('--force', is_flag=True, help='Force save even if configuration file exists')
def save_current_config(ctx, force):
    """Save the current configuration to file."""
    config = ctx.obj['config']
    
    # Determine save path
    home_config_dir = Path.home() / ".config" / "rlhf_ui"
    home_config_dir.mkdir(exist_ok=True, parents=True)
    config_path = home_config_dir / "config.json"
    
    if config_path.exists() and not force:
        click.confirm(f"Configuration file {config_path} already exists. Overwrite?", abort=True)
    
    save_config(config, config_path)
    click.echo(f"Configuration saved to {config_path}")


@cli.command()
@click.pass_context
@click.argument('image_folder', type=click.Path(exists=True, file_okay=False))
@click.option('--count', type=int, default=None, help='Number of images to sample')
def sample_images(ctx, image_folder, count):
    """Sample images from the given folder and display stats."""
    from pathlib import Path
    
    folder = Path(image_folder)
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    images = []
    for ext in extensions:
        images.extend(list(folder.glob(f"*{ext}")))
        images.extend(list(folder.glob(f"*{ext.upper()}")))
    
    if not images:
        click.echo(f"No images found in {folder}")
        return
    
    click.echo(f"Found {len(images)} images in {folder}")
    
    if count and count < len(images):
        import random
        sampled = random.sample(images, count)
        click.echo(f"Sampled {count} images:")
    else:
        sampled = images
        click.echo("All images:")
    
    for img in sampled[:10]:  # Show only first 10
        size = img.stat().st_size / 1024  # KB
        click.echo(f"  - {img.name} ({size:.1f} KB)")
    
    if len(sampled) > 10:
        click.echo(f"  - ... and {len(sampled) - 10} more")


@cli.command()
@click.pass_context
@click.argument('preference_file', type=click.Path(exists=True, dir_okay=False))
def analyze_preferences(ctx, preference_file):
    """Analyze preference data file and show statistics."""
    from rlhf_ui.data.storage import PreferenceDataStorage
    
    # Create temporary storage object
    storage = PreferenceDataStorage(Path(preference_file).parent)
    
    # Get stats
    summary = storage.get_data_summary()
    click.echo(summary)


@cli.command()
@click.pass_context
@click.argument('image1', type=click.Path(exists=True, dir_okay=False))
@click.argument('image2', type=click.Path(exists=True, dir_okay=False))
def compare_images(ctx, image1, image2):
    """Compare two images and calculate similarity."""
    from rlhf_ui.models.embedding import ImageEmbeddingModel
    
    click.echo("Computing image similarity...")
    model = ImageEmbeddingModel()
    
    similarity = model.get_similarity(image1, image2)
    click.echo(f"Similarity score: {similarity:.4f}")
    
    if similarity > 0.9:
        click.echo("Images are very similar")
    elif similarity > 0.7:
        click.echo("Images are moderately similar")
    else:
        click.echo("Images are different")


@cli.command()
@click.pass_context
@click.argument('folder', type=click.Path(exists=True, file_okay=False))
@click.option('--output', '-o', type=click.Path(file_okay=False), 
              help='Output folder for diverse subset')
@click.option('--count', '-n', type=int, default=10, 
              help='Number of diverse images to select')
def select_diverse(ctx, folder, output, count):
    """Select a diverse subset of images from a folder."""
    from rlhf_ui.models.embedding import ImageEmbeddingModel
    import shutil
    
    folder_path = Path(folder)
    if output:
        output_path = Path(output)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Find images
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    images = []
    for ext in extensions:
        images.extend(list(folder_path.glob(f"*{ext}")))
        images.extend(list(folder_path.glob(f"*{ext.upper()}")))
    
    if not images:
        click.echo(f"No images found in {folder}")
        return
    
    if len(images) <= count:
        click.echo(f"Folder contains fewer images ({len(images)}) than requested ({count})")
        selected_images = images
    else:
        click.echo(f"Finding {count} diverse images from {len(images)} total images...")
        
        # Initialize embedding model
        model = ImageEmbeddingModel()
        
        # Select random initial image
        initial_idx = random.randint(0, len(images) - 1)
        
        # Find diverse subset
        selected_indices = model.find_diverse_subset(images, k=count, initial_idx=initial_idx)
        selected_images = [images[i] for i in selected_indices]
    
    # Display or copy selected images
    click.echo(f"Selected {len(selected_images)} diverse images:")
    for i, img in enumerate(selected_images):
        click.echo(f"  {i+1}. {img.name}")
        
        # Copy files if output specified
        if output:
            shutil.copy2(img, output_path / img.name)
    
    if output:
        click.echo(f"Copied selected images to {output_path}")


@cli.command()
@click.pass_context
@click.argument('model_checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.argument('image', type=click.Path(exists=True, dir_okay=False))
@click.option('--prompt', '-p', help='Optional text prompt for context')
def predict_reward(ctx, model_checkpoint, image, prompt):
    """Predict reward for a single image using a trained model."""
    from rlhf_ui.models.trainer import RewardModelTrainer
    
    click.echo(f"Loading model from {model_checkpoint}...")
    
    # Create trainer and load model
    trainer = RewardModelTrainer(
        preference_file=ctx.obj['config'].output_folder / "preferences.csv",
        image_folder=ctx.obj['config'].image_folder,
        model_output_dir=ctx.obj['config'].model_output_dir
    )
    
    try:
        trainer.load_model(model_checkpoint)
        
        # Predict reward
        reward = trainer.predict_reward(image, prompt)
        click.echo(f"Predicted reward: {reward:.6f}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def train(ctx):
    """Train a reward model from the command line."""
    from rlhf_ui.models.trainer import RewardModelTrainer
    
    config = ctx.obj['config']
    preference_file = config.output_folder / "preferences.csv"
    
    if not preference_file.exists():
        click.echo(f"Preference file not found: {preference_file}", err=True)
        sys.exit(1)
    
    click.echo(f"Training reward model using preferences from {preference_file}")
    click.echo(f"Image folder: {config.image_folder}")
    click.echo(f"Output directory: {config.model_output_dir}")
    
    # Create trainer
    trainer = RewardModelTrainer(
        preference_file=preference_file,
        image_folder=config.image_folder,
        model_output_dir=config.model_output_dir
    )
    
    # Setup progress reporting
    with click.progressbar(length=100, label='Training progress') as bar:
        def update_progress(epoch, progress, loss, accuracy):
            bar.update(1)
            if progress == 100:  # End of epoch
                click.echo(f"Epoch {epoch+1}: Loss={loss:.4f}, Accuracy={accuracy:.2%}")
        
        # Train model
        try:
            model_path = trainer.train(
                epochs=config.training_epochs,
                lr=config.learning_rate,
                batch_size=config.batch_size,
                text_embedding_size=768 if config.use_text_embeddings else 0,
                progress_callback=update_progress
            )
            
            click.echo(f"Training completed. Model saved to: {model_path}")
            
            # Export for RLHF
            export_path = trainer.export_for_rlhf()
            click.echo(f"Exported model for RLHF: {export_path}")
            
        except Exception as e:
            click.echo(f"Error during training: {e}", err=True)
            sys.exit(1)


if __name__ == '__main__':
    cli()