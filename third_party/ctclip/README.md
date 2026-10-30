# CT-CLIP image encoder source

`transformer_maskgit/ctvit.py` and `transformer_maskgit/attention.py` are copied
from the authors' [CT-CLIP repository](https://github.com/ibrahimethemhamamci/CT-CLIP),
commit `a2a155c601987820433c01db69b64d701d3d229d`.
The package `__init__.py` here imports only `CTViT`, avoiding unrelated training
dependencies. The authors publish their code under CC BY-NC-SA 4.0; see their
[repository license statement](https://github.com/ibrahimethemhamamci/CT-CLIP#license).

The checkpoint is stored separately at `external/ctclip/CT-CLIP_v2.pt` and is
ignored by Git. `external/ctclip/checkpoint_metadata.json` records its source
and checksum.
