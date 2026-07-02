from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]


def __getattr__(name: str):
    """Load public API objects lazily.

    Importing a nested module such as ``fastvideo.mlx_runtime.fastwan`` executes
    this package ``__init__`` first. Keeping the heavyweight generator/pipeline
    imports lazy lets lightweight runtime tests (notably the Apple Silicon MLX
    smoke job) import only the modules they exercise.
    """
    if name == "VideoGenerator":
        from fastvideo.entrypoints.video_generator import VideoGenerator

        return VideoGenerator
    if name == "PipelineConfig":
        from fastvideo.configs.pipelines import PipelineConfig

        return PipelineConfig
    if name == "SamplingParam":
        from fastvideo.api.sampling_param import SamplingParam

        return SamplingParam
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
