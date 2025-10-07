from importlib.resources import files

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class DataSettings(BaseModel):
    """Settings for data generation."""

    split_ratio: float = 0.8


class ModelSettings(BaseModel):
    """setting for model"""

    cifar10: bool = True
    top_1: bool = True
    input_depth: int = 3
    data_shape: list[int] = [32, 32]
    kernel: list[tuple[int, int]] = [(3, 3), (3, 3)]
    layer_depths: list[int] = [64, 32]
    num_groups: list[int] = [8, 8]
    stride: list[int] = [1, 1]
    l2reg: float = 0.001


class TrainingSettings(BaseModel):
    """Settings for model training."""

    batch_size: int = 128
    num_iters: int = 300
    learning_rate: float = 0.1
    flip_ratio: float = 0.25
    noise_std: float = 0.05
    momentum: float = 0.9


class SaveSettings(BaseModel):
    """Settings for saving"""

    checkpoint_save: str = "/save"
    checkpoint_load: str = "/load"


class AppSettings(BaseSettings):
    """Main application settings."""

    debug: bool = False
    random_seed: int = 31415
    data: DataSettings = DataSettings()
    model: ModelSettings = ModelSettings()
    training: TrainingSettings = TrainingSettings()
    save: SaveSettings = SaveSettings()

    model_config = SettingsConfigDict(
        toml_file=files("hw04").joinpath("config.toml"),
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
