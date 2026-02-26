from __future__ import annotations

from rfstudio.utils.download import download_model_weights

if __name__ == '__main__':
    download_model_weights(
        'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        check_hash=True,
        verbose=True,
    )
