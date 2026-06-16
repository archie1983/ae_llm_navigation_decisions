This project uses LLM models to provide high level guidance for semantic navigation in indoors environments.

If you're deploying on a Jetson, then you will need to build your own BitsAndBytes package: https://forums.developer.nvidia.com/t/bitsandbytes-on-nvidia-jetson-agx-orin/338248/4

But otherwise, just do:

```
git clone https://github.com/archie1983/ae_llm_navigation_decisions
cd ae_llm_navigation_decisions
pip install .
```
Then to try it out launch python and type:

```
from ae_llm_navigation_decisions import LLMDecisionMaker, LLMType
ldm = LLMDecisionMaker(LLMType.MINISTRAL_3_3b_instruct_nf4_bnb)
rt_llm, llm_text = ldm.classify_room_by_this_object_set_and_pic(obj_set={"Scales", "bathtub", "toothbrush"}, img_uri=None)
```