from __future__ import annotations

from rfstudio.data.downloader import BlenderDownloader
from rfstudio.engine.task import TaskGroup

if __name__ == '__main__':
    TaskGroup(
        blender=BlenderDownloader(),
        blender2=BlenderDownloader(),
    ).run()
