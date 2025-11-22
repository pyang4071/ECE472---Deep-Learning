from pathlib import Path
from importlib.resources import files
from typing import Tuple

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    num_features: int = 2
    num_samples: int = 2000
    sigma_noise: float = 0.1
    enc_num_data: int = 10


class ModelSettings(BaseModel):
    """Settings for model"""

    layer_dim: list[int] = [32, 32]
    enc_hidden_dim: int = 4096
    enc_norm_val: float = 0.1


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 128
    num_iters: int = 300
    learning_rate: float = 0.1
    log_clip: float = 1e-7
    l2_loss: float = 1
    enc_num_iters: int = 40000
    enc_batch_size: int = 1024
    enc_lam: int = 5
    enc_lr: float = 0.1


class PlottingSettings(BaseModel):
    """Settings for plotting."""

    figsize: Tuple[int, int] = (5, 3)
    dpi: int = 200
    output_dir: Path = Path("artifacts")
    enc_dpi: int = 300
    enc_features: list[int] = [1, 2, 3]


class SaveSettings(BaseModel):
    """Settings for saving"""

    checkpoint_save_mlp: str = "/save_mlp"
    checkpoint_load_mlp: str = "/load_mlp"
    checkpoint_save_enc: str = "/save_enc"
    checkpoint_load_enc: str = "/load_enc"


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()
    save: SaveSettings = SaveSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw07").joinpath("config.toml"),
        env_nested_delimiter="__",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Set the priority of settings sources.

        We use a TOML file for configuration.
        """
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
