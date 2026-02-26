# **Installation**

## **Prerequisites**

Radiance Field Studio requires `python >= 3.10` . We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

## **Setup Conda Environment**

```pshell
conda create --name rfstudio -y python=3.10
conda activate rfstudio
pip install --upgrade pip setuptools
```

## **Install PyTorch**

```pshell
pip install numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2
python -c "import torch; torch.zeros(1).cuda()" || echo "ERROR: CUDA check failed"
```

## **Install Components from NVlabs**

```pshell
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
```

## **Install Codebase**

=== "SSH"

    ```pshell
    git clone git@github.com:PKU-VCL-Geometry/RadianceFieldStudio.git
    cd RadianceFieldStudio
    pip install -e .
    ```

=== "HTTPS"

    ```pshell
    git clone https://github.com/PKU-VCL-Geometry/RadianceFieldStudio.git
    cd RadianceFieldStudio
    pip install -e .
    ```

## **Setup Editor Extensions [Optional]**

=== "VSCode"

    + [Ext:Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

        Support IntelliSense (Pylance) and code navigation.

    + [Ext:Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

        Support auto-imports.

    + [Ext:Yapf](https://marketplace.visualstudio.com/items?itemName=eeyore.yapf)

        Support code formatting.
