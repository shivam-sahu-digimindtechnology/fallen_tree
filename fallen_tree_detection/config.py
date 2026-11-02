from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    MODEL_PATH: str = r"F:\fallen_tree\fallen_tree_detection\Tree_detection.pt"

    # Video
    VIDEO_SOURCE: str | int = r"F:\fallen_tree\fallen_tree_detection\data\treesvdo.mp4"
    # VIDEO_SOURCE: str | int = 0   # uncomment for webcam

    # Detection
    CONFIDENCE_THRESHOLD: float = 0.5

    # Display
    SHOW_WINDOW: bool = True
    PRINT_DETECTIONS: bool = True
    SAVE_OUTPUT: bool = True
    OUTPUT_PATH: str = r"F:\fallen_tree\fallen_tree_detection\result.mp4"


settings = Settings()