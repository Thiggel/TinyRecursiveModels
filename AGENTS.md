Developer notes for this repo
-----------------------------

- Use the Apptainer image when running Python or PyTorch CLI tools. Example:
  - `apptainer exec --nv --pwd /home/hpc/c107fa/c107fa12/TinyRecursiveModels /home/atuin/c107fa/c107fa12/TinyRecursiveModels/containers/pytorch.sif python your_script.py`
  - Adjust the command after `python` as needed (e.g., `-m py_compile pretrain.py`).
- The repository lives at `/home/hpc/c107fa/c107fa12/TinyRecursiveModels`; the container image is at `/home/atuin/c107fa/c107fa12/TinyRecursiveModels/containers/pytorch.sif`.
