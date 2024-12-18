## 📘 **Theory**
Fine-tuning builds on the foundation of transfer learning. By adapting a pre-trained LLM to a 
specific domain or task, you achieve higher accuracy and relevance without the need to train 
from scratch. Fine-tuning involves adjusting the model's weights based on a smaller dataset 
that’s specific to the task, improving domain-specific performance.
Some of the fine-tuning methods are: Supervised Fine-Tuning (SFT) and Reward Modelling (RLHF and DPO)


### Full Training vs. Low-Rank Adaptation (LoRA) Training
When fine-tuning an LLM, there are two approaches:
- **Full Training**: Involves updating all model parameters, which can be computationally expensive and time-consuming, especially for large models.
- **LoRA (Low-Rank Adaptation)**: A more efficient method that updates only a smaller subset of model parameters, reducing computational cost and training time. LoRA allows for the fine-tuning of large models with fewer resources by adjusting a smaller, low-rank matrix while keeping the rest of the model frozen.


### Supervised Fine-Tuning (SFT)
Supervised Fine-Tuning (SFT) involves training the model on labeled data, where both the 
inputs and expected outputs are provided. This allows the model to improve on tasks such 
as text classification, named entity recognition, or question answering by learning patterns
and correlations directly from the data.

### Reward Modelling: RLHF and DPO
Reward modeling helps optimize LLMs by rewarding desired behaviors and penalizing incorrect 
or undesired outcomes. Two popular approaches are:
- **Reinforcement Learning from Human Feedback (RLHF)**: This technique improves models by 
incorporating human feedback during training, allowing the model to optimize for preferred 
outcomes.
- **Direct Preference Optimization (DPO)**: DPO also uses human preferences to fine-tune 
the model but focuses on a more direct optimization of user-desired outcomes rather than 
relying solely on trial and error.

## 🚀 **Decide which approach to use**
Of course, there isn't a solution that fits it all. It all depends on the specific use case.
The goal of this repo is to provide one or more approaches for each use case we mention.
- [**Improve chatbot accuracy**](https://github.com/gen-mind/fine-tuning/tree/main/usecase-chatbot/readme.md)   we will use  a Supervised Fine-Tuning (SFT) approach
- **Understand industry-specific language**  we will use  a Direct Preference Optimization (DPO)*
- **Diagnose technical or mechanical issues**  
- **Assess medical conditions**  
- **Evaluate and estimate car damage**  
- **Analyze legal documents**  
- **Detect patterns in images**  
  - **Identify objects or specific features**  
  - **Evaluate product defects or manufacturing inconsistencies**  
  - **Identify damages that require maintenance**  
  - **Identify abnormalities in medical scans**  
- **Assist in scientific research**  


## 📑 References:
- **Hugging Face Fine-Tuning Concepts**: [Fine-Tuning Language Models with Hugging Face](https://huggingface.co/blog/fine-tune-transformers)
- **Transfer Learning and Fine-Tuning Guide**: [Google’s Transfer Learning Documentation](https://developers.google.com/machine-learning/glossary#transfer_learning)
- **Hugging Face SFT - RLHF and DPO**: [Fine-Tuning LLMs: Supervised Fine-Tuning and Reward Modelling](https://huggingface.co/blog/rishiraj/finetune-llms)
- **RLHF Overview**: [OpenAI’s Blog on Training GPT-3](https://openai.com/blog/instruction-following)



## **Additional fine-tuning methods**
https://www.turing.com/resources/finetuning-large-language-models

Fine-tuning is the process of adjusting the parameters of a pre-trained large language model to a specific task or domain. Although pre-trained language models like GPT possess vast language knowledge, they lack specialization in specific areas. Fine-tuning addresses this limitation by allowing the model to learn from domain-specific data to make it more accurate and effective for targeted applications.

By exposing the model to task-specific examples during fine-tuning, the model can acquire a deeper understanding of the nuances of the domain. This bridges the gap between a general-purpose language model and a specialized one, unlocking the full potential of LLMs in specific domains or applications.

- **Supervised Fine-Tuning (SFT)**
   - **SFT** is the process of updating a pre-trained model's parameters by training it on labeled data for the specific task you want the model to perform. For instance, you can fine-tune a model to be better at summarization by training it with examples of text and their summaries.
   - It’s useful for domains where **labeled examples** exist, such as Q&A datasets or chat logs.
   - Learn more: [Hugging Face Transformers Fine-Tuning Tutorial](https://huggingface.co/course/chapter3/3)

- **Reward Modelling: RLHF and DPO**
   - **Reinforcement Learning with Human Feedback (RLHF)** adds a feedback loop where human reviewers rate the model’s outputs, which then helps fine-tune the model to generate better responses. This is useful for **conversational AI** because it improves the quality of the interaction in response to human evaluations.
   - **Direct Preference Optimization (DPO)** is a simpler version of RLHF that avoids explicitly training a reward model, directly optimizing based on preferences. This is aimed at reducing complexity while still leveraging human feedback.
   - Learn more about RLHF: [OpenAI's Guide to Reinforcement Learning](https://openai.com/research/reinforcement-learning-from-human-feedback)
   - For an overview of DPO: [Anthropic’s Direct Preference Optimization](https://www.anthropic.com/blog/dpo-preference-model)

- **Prefix-Tuning**
   - This method fine-tunes a model by learning task-specific prefixes that influence how the LLM processes input and generates output. It keeps the core model weights frozen, making it more memory efficient. It’s particularly useful for generative tasks like text completion and dialog.
   - Learn more: [Prefix-Tuning](https://arxiv.org/abs/2101.00190)
- **Prompt-Tuning**
   - Prompt-tuning focuses on learning a set of prompt tokens that are prepended to the input, effectively guiding the model to produce domain-specific responses. This is also efficient in terms of memory and computational resources since only a small number of parameters are updated.
   - Learn more: [Prompt Tuning for GPT-3](https://arxiv.org/abs/2104.08691)

- **Adapter Tuning**
   - Adapters are small bottleneck layers inserted into each layer of a transformer. Only these adapter layers are trained during fine-tuning, keeping the majority of the model frozen. This method is resource-efficient and is particularly advantageous when multiple tasks are involved, as different adapters can be used with the same core model.
   - Learn more: [Adapters in Transformers](https://arxiv.org/abs/1902.00751)

- **Parameter Efficient Fine-Tuning (PEFT)**
  - PEFT methods, like adapters and LoRA, aim to fine-tune a small subset of model parameters. These methods are especially valuable for handling large LLMs on resource-limited setups. They reduce the memory footprint significantly while maintaining performance.
  - Learn more: [Hugging Face PEFT Methods](https://huggingface.co/docs/peft/index)
- **Distillation-Based Fine-Tuning**
   - This involves a process called **Knowledge Distillation**, where a smaller "student" model learns to mimic a larger, fine-tuned "teacher" model. It allows for efficiency gains by producing a more compact version of a high-performing model.
   - Learn more: [DistilBERT: A distilled version of BERT](https://arxiv.org/abs/1910.01108)


## **Full Training vs. Low-Rank Adaptation (LoRA)**
   - **Full Training** means updating all the parameters of a model. This is computationally expensive and often unnecessary for fine-tuning, especially with LLMs.
   - **LoRA (Low-Rank Adaptation)** reduces computational requirements by updating only a smaller number of model weights, which is highly effective for tasks that don't need all of the model's capacity.
   - For more details on LoRA: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)



- **Full Training**:
  - **Full training** means updating *all* the parameters of the model during the training process.
  - In the context of a pre-trained language model, this means the entire model, with millions or even billions of parameters, is adjusted during fine-tuning. 
  - Full training is **resource-intensive**—it requires a lot of **compute power**, **memory**, and **time** because you need to backpropagate gradients through all of the model’s layers.
  - Typically, full training is done when you train a model from scratch, but it can also be done to adapt a model to a new task or dataset completely. However, this can be **impractical** for very large models like GPT-3 or LLaMA, as it demands enormous computational resources.

- **Low-Rank Adaptation (LoRA)**:
  - **LoRA** is a **parameter-efficient** fine-tuning approach where, instead of updating all the parameters of the model, you update a small subset of them by adding new "low-rank" matrices into each layer.
  - The idea is to decompose the change in weights into smaller rank matrices. The original model weights remain **frozen** (unchanged), and the adjustments are captured through these additional, lower-dimensional matrices.
  - This approach **reduces resource requirements**, meaning you can fine-tune a large model with less GPU memory, and it's suitable when you need to adapt a model to multiple different tasks, since you can use different LoRA layers for different tasks while maintaining the same base model.

### Supervised Fine-Tuning (SFT)

**Supervised Fine-Tuning (SFT)** is a type of fine-tuning where you use a **labeled dataset** to guide the model's learning. The goal is to make the model perform better on a specific task, such as summarization or question-answering, by training it with labeled pairs of inputs and expected outputs.

- **SFT can be done with both Full Training and LoRA**:
  - **With Full Training**: You perform SFT by updating **all** the model parameters. This is very effective but can be **expensive**, especially for large LLMs.
  - **With LoRA**: You perform SFT by updating only a small number of **LoRA parameters** (additional matrices) while keeping the main model frozen. This is much more efficient and often yields good results without needing the massive compute power of full training.

### Summary of the Relationship

- **Full Training** updates **all** parameters during SFT, which is computationally heavy but can lead to significant adaptation of the model.
- **LoRA** is a **more efficient way** to do SFT by only updating a small number of additional parameters, making it easier and cheaper to adapt large models.


- You can **use SFT with either method**—LoRA or Full Training. It’s just about how many parameters you decide to adjust:
  - If you have the resources, full training can be applied.
  - If you need efficiency and want to avoid retraining the entire model, LoRA is the better approach.

So, **SFT** is not inherently tied to LoRA or Full Training; it’s about using a labeled dataset, and **how you train the model** depends on your resources and goals.