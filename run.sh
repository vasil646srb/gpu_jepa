root@O-1960519:~# https://github.com/vasil646srb/gpu_jepa.git
-bash: https://github.com/vasil646srb/gpu_jepa.git: No such file or directory
root@O-1960519:~# git clone https://github.com/vasil646srb/gpu_jepa.git
Cloning into 'gpu_jepa'...
remote: Enumerating objects: 71, done.
remote: Counting objects: 100% (71/71), done.
remote: Compressing objects: 100% (50/50), done.
remote: Total 71 (delta 38), reused 50 (delta 20), pack-reused 0 (from 0)
Receiving objects: 100% (71/71), 55.27 KiB | 1.54 MiB/s, done.
Resolving deltas: 100% (38/38), done.
root@O-1960519:~# cd gpu_jepa/
root@O-1960519:~/gpu_jepa# bash run.sh 
==========================================
🚀 GPU-JEPA Deployment Script
Container: nvcr.io/nvidia/pytorch:24.08-py3
==========================================

🔍 Проверка окружения...
Python 3.10.12

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.10/dist-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/usr/local/lib/python3.10/dist-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/usr/local/lib/python3.10/dist-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/usr/local/lib/python3.10/dist-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
PyTorch: 2.5.0a0+872d972e41.nv24.08
CUDA available: True
CUDA version: 12.6

📦 Обновление pip...
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: pip in /usr/local/lib/python3.10/dist-packages (26.1.2)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (70.3.0)
Collecting setuptools
  Downloading setuptools-83.0.0-py3-none-any.whl.metadata (6.6 kB)
Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (0.44.0)
Collecting wheel
  Downloading wheel-0.47.0-py3-none-any.whl.metadata (2.3 kB)
Collecting packaging>=24.0 (from wheel)
  Downloading packaging-26.2-py3-none-any.whl.metadata (3.5 kB)
Downloading setuptools-83.0.0-py3-none-any.whl (1.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 18.9 MB/s  0:00:00
Downloading wheel-0.47.0-py3-none-any.whl (32 kB)
Downloading packaging-26.2-py3-none-any.whl (100 kB)
Installing collected packages: setuptools, packaging, wheel
  Attempting uninstall: setuptools
    Found existing installation: setuptools 70.3.0
    Uninstalling setuptools-70.3.0:
      Successfully uninstalled setuptools-70.3.0
  Attempting uninstall: packaging
    Found existing installation: packaging 23.2
    Uninstalling packaging-23.2:
      Successfully uninstalled packaging-23.2
  Attempting uninstall: wheel
    Found existing installation: wheel 0.44.0
    Uninstalling wheel-0.44.0:
      Successfully uninstalled wheel-0.44.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
cudf 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
cudf 24.6.0 requires pyarrow<16.2.0a0,>=16.1.0, but you have pyarrow 25.0.0 which is incompatible.
cugraph 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
cugraph-dgl 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
cugraph-pyg 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
cugraph-service-server 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
dask-cudf 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
kvikio 24.6.0 requires numpy<2.0a0,>=1.23, but you have numpy 2.2.6 which is incompatible.
nvidia-modelopt 0.15.0 requires numpy<2, but you have numpy 2.2.6 which is incompatible.
thinc 8.2.5 requires numpy<2.0.0,>=1.19.0; python_version >= "3.9", but you have numpy 2.2.6 which is incompatible.
Successfully installed packaging-26.2 setuptools-83.0.0 wheel-0.47.0
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.

📦 Установка зависимостей...
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: sentence-transformers in /usr/local/lib/python3.10/dist-packages (5.6.0)
Requirement already satisfied: transformers in /usr/local/lib/python3.10/dist-packages (4.51.3)
Requirement already satisfied: accelerate in /usr/local/lib/python3.10/dist-packages (1.14.0)
Requirement already satisfied: huggingface-hub in /usr/local/lib/python3.10/dist-packages (0.36.2)
Requirement already satisfied: pyarrow in /usr/local/lib/python3.10/dist-packages (25.0.0)
Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (2.2.6)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (1.14.0)
Requirement already satisfied: torch>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from sentence-transformers) (2.5.0a0+872d972e41.nv24.8)
Requirement already satisfied: scikit-learn>=0.22.0 in /usr/local/lib/python3.10/dist-packages (from sentence-transformers) (1.5.1)
Requirement already satisfied: typing_extensions>=4.5.0 in /usr/local/lib/python3.10/dist-packages (from sentence-transformers) (4.12.2)
Requirement already satisfied: tqdm>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from sentence-transformers) (4.66.5)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers) (3.15.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers) (26.2)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers) (2026.7.10)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers) (2.32.3)
Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.21.4)
Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.10/dist-packages (from transformers) (0.8.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub) (2024.6.1)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub) (1.5.2)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate) (6.0.0)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.22.0->sentence-transformers) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.22.0->sentence-transformers) (3.5.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence-transformers) (3.3)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence-transformers) (3.1.4)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.11.0->sentence-transformers) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch>=1.11.0->sentence-transformers) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.11.0->sentence-transformers) (2.1.5)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->transformers) (2024.7.4)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting onnxruntime-gpu
  Downloading onnxruntime_gpu-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (5.4 kB)
Collecting coloredlogs (from onnxruntime-gpu)
  Downloading coloredlogs-15.0.1-py2.py3-none-any.whl.metadata (12 kB)
Collecting flatbuffers (from onnxruntime-gpu)
  Downloading flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
Requirement already satisfied: numpy>=1.21.6 in /usr/local/lib/python3.10/dist-packages (from onnxruntime-gpu) (2.2.6)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from onnxruntime-gpu) (26.2)
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from onnxruntime-gpu) (4.24.4)
Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from onnxruntime-gpu) (1.13.1)
Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime-gpu)
  Downloading humanfriendly-10.0-py2.py3-none-any.whl.metadata (9.2 kB)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->onnxruntime-gpu) (1.3.0)
Downloading onnxruntime_gpu-1.23.2-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (300.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 300.5/300.5 MB 108.0 MB/s  0:00:02
Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
Downloading flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Installing collected packages: flatbuffers, humanfriendly, coloredlogs, onnxruntime-gpu
Successfully installed coloredlogs-15.0.1 flatbuffers-25.12.19 humanfriendly-10.0 onnxruntime-gpu-1.23.2
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: FlagEmbedding in /usr/local/lib/python3.10/dist-packages (1.4.0)
Requirement already satisfied: torch>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (2.5.0a0+872d972e41.nv24.8)
Requirement already satisfied: transformers<6.0.0,>=4.44.2 in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (4.51.3)
Requirement already satisfied: datasets>=2.19.0 in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (5.0.0)
Requirement already satisfied: accelerate>=0.20.1 in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (1.14.0)
Requirement already satisfied: sentence_transformers in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (5.6.0)
Requirement already satisfied: peft in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (0.19.1)
Requirement already satisfied: ir-datasets in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (0.6.2)
Requirement already satisfied: sentencepiece in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (0.2.2)
Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from FlagEmbedding) (4.24.4)
Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (3.15.4)
Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (0.36.2)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (2.2.6)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (26.2)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (6.0.2)
Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (2026.7.10)
Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (2.32.3)
Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (0.21.4)
Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (0.8.0)
Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.10/dist-packages (from transformers<6.0.0,>=4.44.2->FlagEmbedding) (4.66.5)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers<6.0.0,>=4.44.2->FlagEmbedding) (2024.6.1)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers<6.0.0,>=4.44.2->FlagEmbedding) (1.5.2)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers<6.0.0,>=4.44.2->FlagEmbedding) (4.12.2)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from accelerate>=0.20.1->FlagEmbedding) (6.0.0)
Requirement already satisfied: pyarrow>=21.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (25.0.0)
Requirement already satisfied: dill<0.4.2,>=0.3.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (0.4.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (2.2.2)
Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (0.27.0)
Requirement already satisfied: xxhash in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (3.8.1)
Requirement already satisfied: multiprocess<0.70.20 in /usr/local/lib/python3.10/dist-packages (from datasets>=2.19.0->FlagEmbedding) (0.70.19)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.10/dist-packages (from fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (3.10.1)
Requirement already satisfied: anyio in /usr/local/lib/python3.10/dist-packages (from httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (4.4.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (2024.7.4)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (1.0.5)
Requirement already satisfied: idna in /usr/local/lib/python3.10/dist-packages (from httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (3.7)
Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (1.3.1)
Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (0.14.0)
Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (2.3.5)
Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (1.3.1)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (24.2.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (1.4.1)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (6.0.5)
Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (1.9.4)
Requirement already satisfied: async-timeout<5.0,>=4.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.19.0->FlagEmbedding) (4.0.3)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->transformers<6.0.0,>=4.44.2->FlagEmbedding) (3.3.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->transformers<6.0.0,>=4.44.2->FlagEmbedding) (2.0.7)
Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.6.0->FlagEmbedding) (3.3)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.6.0->FlagEmbedding) (3.1.4)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.10/dist-packages (from torch>=1.6.0->FlagEmbedding) (1.13.1)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy==1.13.1->torch>=1.6.0->FlagEmbedding) (1.3.0)
Requirement already satisfied: exceptiongroup>=1.0.2 in /usr/local/lib/python3.10/dist-packages (from anyio->httpx<1.0.0->datasets>=2.19.0->FlagEmbedding) (1.2.2)
Requirement already satisfied: lxml>=4.5.2 in /usr/local/lib/python3.10/dist-packages (from ir-datasets->FlagEmbedding) (6.1.1)
Requirement already satisfied: lz4>=3.1.10 in /usr/local/lib/python3.10/dist-packages (from ir-datasets->FlagEmbedding) (4.4.5)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.6.0->FlagEmbedding) (2.1.5)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets>=2.19.0->FlagEmbedding) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets>=2.19.0->FlagEmbedding) (2023.4)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.10/dist-packages (from pandas->datasets>=2.19.0->FlagEmbedding) (2024.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas->datasets>=2.19.0->FlagEmbedding) (1.16.0)
Requirement already satisfied: scikit-learn>=0.22.0 in /usr/local/lib/python3.10/dist-packages (from sentence_transformers->FlagEmbedding) (1.5.1)
Requirement already satisfied: scipy>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from sentence_transformers->FlagEmbedding) (1.14.0)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.22.0->sentence_transformers->FlagEmbedding) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.22.0->sentence_transformers->FlagEmbedding) (3.5.0)
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.

✅ Проверка установленных пакетов...

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<string>", line 2, in <module>
  File "/usr/local/lib/python3.10/dist-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/usr/local/lib/python3.10/dist-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/usr/local/lib/python3.10/dist-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/usr/local/lib/python3.10/dist-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_worker/__main__.py", line 8, in <module>
    from torch._inductor.async_compile import pre_fork_setup
  File "/usr/local/lib/python3.10/dist-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/usr/local/lib/python3.10/dist-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/usr/local/lib/python3.10/dist-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/usr/local/lib/python3.10/dist-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
✅ torch: 2.5.0a0+872d972e41.nv24.08
✅ transformers: 4.51.3
✅ sentence-transformers: 5.6.0
✅ onnxruntime: 1.23.2
✅ pyarrow: 25.0.0
✅ numpy: 2.2.6
✅ ONNX providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
✅ GPU: NVIDIA GeForce RTX 3090
✅ VRAM: 25.3 GB

📁 Создание структуры проекта...

📂 Копирование исходников...
⚠️  Исходники не найдены рядом со скриптом.
   Ожидается структура:
   run.sh
   config.py
   train_streaming.py
   test_jepa.py
   boss_fight.py
   models.py
   dataset.py
   losses.py

🔍 Проверка исходников...
  ❌ config.py — отсутствует!
  ❌ models.py — отсутствует!
  ❌ dataset.py — отсутствует!
  ❌ losses.py — отсутствует!
  ❌ train_streaming.py — отсутствует!
  ❌ test_jepa.py — отсутствует!
  ❌ boss_fight.py — отсутствует!

🔍 Проверка синтаксиса Python...
[Errno 2] No such file or directory: '/workspace/gpu_jepa/config.py'root@O-1960519:~/gpu_jepa# c^C
root@O-1960519:~/gpu_jepa# ^C
root@O-1960519:~/gpu_jepa# ls
LICENSE  boss_fight.py  config.py  dataset.py  losses.py  models.py  run.sh  test_jepa.py  train_streaming.py  upload_to_hf.py
root@O-1960519:~/gpu_jepa# ls -la
total 128
drwxr-xr-x 3 root root  4096 Jul 17 16:05 .
drwx------ 1 root root   124 Jul 17 16:05 ..
drwxr-xr-x 8 root root   163 Jul 17 16:05 .git
-rw-r--r-- 1 root root   113 Jul 17 16:05 .gitignore
-rw-r--r-- 1 root root 11357 Jul 17 16:05 LICENSE
-rw-r--r-- 1 root root  8254 Jul 17 16:05 boss_fight.py
-rw-r--r-- 1 root root 15911 Jul 17 16:05 config.py
-rw-r--r-- 1 root root  1788 Jul 17 16:05 dataset.py
-rw-r--r-- 1 root root  5884 Jul 17 16:05 losses.py
-rw-r--r-- 1 root root  5536 Jul 17 16:05 models.py
-rwxr-xr-x 1 root root  7880 Jul 17 16:05 run.sh
-rw-r--r-- 1 root root 24183 Jul 17 16:05 test_jepa.py
-rw-r--r-- 1 root root 21541 Jul 17 16:05 train_streaming.py
-rw-r--r-- 1 root root  1051 Jul 17 16:05 upload_to_hf.py
root@O-1960519:~/gpu_jepa# ^C
root@O-1960519:~/gpu_jepa# ^C
root@O-1960519:~/gpu_jepa# python train_streaming.py --num-files 1 --examples-per-file 100

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/root/gpu_jepa/train_streaming.py", line 15, in <module>
    import torch
  File "/usr/local/lib/python3.10/dist-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/usr/local/lib/python3.10/dist-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/usr/local/lib/python3.10/dist-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/usr/local/lib/python3.10/dist-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
🔥 Обнаружена GPU: NVIDIA GeForce RTX 3090
   VRAM: 25.3 GB
🎯 Режим: GPU (CUDA)
🎯 Конфигурация:
   Модель: Qwen/Qwen3-Embedding-0.6B
   Файлов: 1
   Примеров/файл: 100
   Шагов/шард: 1000
   Всего примеров: 100
   Всего шагов: 1000
======================================================================
🚀 STREAMING ОБУЧЕНИЕ Text-JEPA
======================================================================
📁 Файлов для обработки: 1
📊 Примеров с каждого файла: 100
🔄 Шагов обучения на шард: 1000
💻 Устройство: cuda
   GPU: NVIDIA GeForce RTX 3090
   VRAM: 25.3 GB
📦 Загрузка embedding модели: Qwen/Qwen3-Embedding-0.6B

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/usr/local/lib/python3.10/dist-packages/torch/_inductor/compile_worker/__main__.py", line 8, in <module>
    from torch._inductor.async_compile import pre_fork_setup
  File "/usr/local/lib/python3.10/dist-packages/torch/__init__.py", line 2422, in <module>
    from torch import (
  File "/usr/local/lib/python3.10/dist-packages/torch/export/__init__.py", line 64, in <module>
    from .dynamic_shapes import Constraint, Dim, dims, dynamic_dim, ShapesCollection
  File "/usr/local/lib/python3.10/dist-packages/torch/export/dynamic_shapes.py", line 18, in <module>
    from .exported_program import ExportedProgram
  File "/usr/local/lib/python3.10/dist-packages/torch/export/exported_program.py", line 24, in <module>
    from torch._higher_order_ops.utils import autograd_not_implemented
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/__init__.py", line 1, in <module>
    from torch._higher_order_ops.cond import cond
  File "/usr/local/lib/python3.10/dist-packages/torch/_higher_order_ops/cond.py", line 5, in <module>
    import torch._subclasses.functional_tensor
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 42, in <module>
    class FunctionalTensor(torch.Tensor):
  File "/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py", line 267, in FunctionalTensor
    cpu = _conversion_method_template(device=torch.device("cpu"))
/usr/local/lib/python3.10/dist-packages/torch/_subclasses/functional_tensor.py:267: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /opt/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  cpu = _conversion_method_template(device=torch.device("cpu"))
   🎯 Backend: Sentence-Transformers
Traceback (most recent call last):
  File "/root/gpu_jepa/train_streaming.py", line 513, in <module>
    main()
  File "/root/gpu_jepa/train_streaming.py", line 506, in main
    streaming_train(
  File "/root/gpu_jepa/train_streaming.py", line 331, in streaming_train
    engine = EmbeddingEngine(config.model_path, device=DEVICE.type)
  File "/root/gpu_jepa/train_streaming.py", line 69, in __init__
    self.dim = self._detect_dim()
  File "/root/gpu_jepa/train_streaming.py", line 87, in _detect_dim
    test_emb = self.encode(["test sentence"], batch_size=1)
  File "/root/gpu_jepa/train_streaming.py", line 121, in encode
    return torch.cat(all_embs, dim=0).float().numpy()
RuntimeError: Numpy is not available
root@O-1960519:~/gpu_jepa# 

