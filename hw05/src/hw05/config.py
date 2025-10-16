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

    max_token: int = 10


class ModelSettings(BaseModel):
    """setting for model"""

    token_length: int = 1
    num_classes: int = 4
    num_layers: int = 4
    hidden_layer_width: int = 6
    embed_depth: int = 128


class TrainingSettings(BaseModel):
    """Settings for model training."""

    num_fold: int = 1
    learning_rate: float = 0.1
    batch_size: int = 128
    num_iters: int = 300
    l2reg: float = 9000.0


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
        toml_file=files("hw05").joinpath("config.toml"),
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
