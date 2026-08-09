from dataclasses import dataclass
import platform
from pathlib import Path

os_name = platform.system()


@dataclass(frozen=True)
class Pathes:
    if os_name == "Windows":
        wsf = Path(r"E:\ProjectLocal\Robotics\ACT++")
    elif os_name == "Linux":
        wsf = Path("/media/fhz/Learning/ProjectLocal/Robotics/ACT++")
    data_dir = Path("E:\Dataset\Aloha")
    xml_dir = wsf / "assets"
    config_dir = wsf / "config"
    output_dir = wsf / "output"
    ckpt_dir = wsf / "ckpt"


P = Pathes()
