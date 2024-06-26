.. _models_llm_falcon-instruct:

========================================
falcon-instruct
========================================

- **Context Length:** 2048
- **Model Name:** falcon-instruct
- **Languages:** en
- **Abilities:** chat
- **Description:** Falcon-instruct is a fine-tuned version of the Falcon LLM, specializing in chatting.

Specifications
^^^^^^^^^^^^^^


Model Spec 1 (pytorch, 7 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 7
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** tiiuae/falcon-7b-instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tiiuae/falcon-7b-instruct>`__, `ModelScope <https://modelscope.cn/models/Xorbits/falcon-7b-instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name falcon-instruct --size-in-billions 7 --model-format pytorch --quantization ${quantization}


Model Spec 2 (pytorch, 40 Billion)
++++++++++++++++++++++++++++++++++++++++

- **Model Format:** pytorch
- **Model Size (in billions):** 40
- **Quantizations:** 4-bit, 8-bit, none
- **Engines**: Transformers
- **Model ID:** tiiuae/falcon-40b-instruct
- **Model Hubs**:  `Hugging Face <https://huggingface.co/tiiuae/falcon-40b-instruct>`__

Execute the following command to launch the model, remember to replace ``${quantization}`` with your
chosen quantization method from the options listed above::

   xinference launch --model-engine ${engine} --model-name falcon-instruct --size-in-billions 40 --model-format pytorch --quantization ${quantization}

