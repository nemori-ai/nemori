# Nemori：自然启发的情景记忆系统  
Nemori: Nature-Inspired Episodic Memory System

## 项目概述  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/nemori-ai/nemori)
## Project Overview

Nemori-AI 旨在让大语言模型具备类人的情景记忆能力。  
Nemori-AI empowers large language models with human-like episodic memory.

Nemori 通过自然、事件化的索引方式，帮助系统在关键时刻精准回溯原始经历。  
Nemori stores experiences as natural, event-centric traces, enabling precise recall when it matters.

愿景：让每一次数据交互，都能像人类记忆一样被理解、被回忆、被延续。  
**Vision:** Every piece of data remembered and retrieved as intuitively as human recollection.

Nemori 源自我们团队 Tanka.ai 项目中记忆系统的情景记忆索引模块——这是一个我们计划开源的 MVP 实现。其核心目标是分享我们通过自然启发的情景记忆来构建记忆索引（Nemori: Nature-Inspired Episodic Memory）的方法。  
Nemori is derived from our team's episodic memory indexing module within the memory system of our Tanka.ai project—an MVP implementation that we plan to open-source. Its core purpose is to share our approach to building memory indexing through Nature-Inspired Episodic Memory.

虽然 Mem0、Letta、Supermemory、ZEP、MemOS 等先前系统在 AI 记忆方面做出了卓越尝试，在 LoCoMo 基准测试中取得了先进的性能，但 Nemori 引入了一种创新且简约的方法，专注于与人类情景记忆模式保持一致。鉴于最近优秀的开源项目和记忆系统研究的涌现，我们都汇聚在使用 LoCoMo 数据集作为基准，因此我们决定用展示我们情景记忆索引方法的 MVP 实现参与这一基准测试。  
While previous systems like Mem0, Letta, Supermemory, ZEP, and MemOS have made remarkable attempts at AI memory, achieving advanced performance on the LoCoMo benchmark, Nemori introduces an innovative and minimalist approach centered on aligning with human episodic memory patterns. Given the recent surge of excellent open-source projects and research in memory systems, we've all converged on using the LoCoMo dataset as a benchmark. Consequently, we decided to participate in this benchmark with our MVP implementation that demonstrates our episodic memory indexing approach.

## 实验结果  
## Experimental Results

为了突出 Nemori 这个简洁方法的优越性，我们在 LoCoMo 基准测试上进行了评估，与先前的最先进方法进行了比较：  
To highlight the superiority of Nemori's concise approach, we conducted evaluations on the LoCoMo benchmark, comparing against previous state-of-the-art approaches:

### LoCoMo 基准测试结果  
### LoCoMo Benchmark Results

在 LoCoMo（长上下文对话建模）数据集上，Nemori 展现了卓越的性能：  
On the LoCoMo (Long-Context Conversation Modeling) dataset, Nemori demonstrates exceptional performance:

![LoCoMo Benchmark Results](figures/locomo-scores.png)

## 复现方法（Reproduction Guide）  
## Reproduction Guide

要复现 Nemori 在 LoCoMo 基准测试上的实验结果，请参考 [evaluation/README.md](evaluation/README.md) 获取详细的评测环境搭建与运行步骤。  
To reproduce Nemori's experimental results on the LoCoMo benchmark, please refer to [evaluation/README.md](evaluation/README.md) for detailed evaluation environment setup and execution steps.

## 设计理念  
## Design Philosophy

当我们人类回忆过去的事件时，我们的脑海中经常闪现相关的图像、动作或声音。我们的大脑通过让我们重新体验当时发生的事情来帮助我们记忆——这种记忆机制被称为情景记忆。  
When we humans recall past events, our minds often flash with related images, actions, or sounds. Our brains help us remember by essentially making us re-experience what happened at that time - this memory mechanism is called episodic memory.

Nemori 的设计灵感来自人类的情景记忆。Nemori 可以自主地将人与人、人与 AI 智能体或 AI 智能体之间的对话重塑为情景片段。与原始对话相比，情景片段具有更连贯的因果关系和时间表达能力。更重要的是，情景片段的表达在某种程度上与我们人类记忆回忆的粒度相一致，这意味着作为人类，我们更可能提出关于情景片段的问题，这些问题在语义上更接近情景片段本身，而不是原始消息。  
Nemori's design inspiration comes from human episodic memory. Nemori can autonomously reshape conversations between humans, between humans and AI agents, or between AI agents into episodes. Compared to raw conversations, episodes have more coherent causal relationships and temporal expression capabilities. More importantly, the expression of episodes aligns to some extent with the granularity of our human memory recall, meaning that as humans, we are likely to ask questions about episodes that are semantically closer to the episodes themselves rather than the original messages.

### 颗粒度与大模型训练分布的对齐  
### Granularity Alignment with LLM Training Distribution

我们设计中的一个关键洞察是，情景记忆颗粒度对齐为大语言模型提供了潜在的优化收益。由于大模型的训练数据集会对齐人类世界的文本分布，所以对齐回忆颗粒度的同时，也在对齐「自然世界中最大概率的事件表述颗粒度」。  
A key insight in our design is that episodic memory granularity alignment offers potential optimization benefits for large language models. Since LLM training datasets align with the textual distribution of the human world, aligning recall granularity simultaneously aligns with the "most probable event description granularity in the natural world."

这种对齐提供了几个优势：  
This alignment provides several advantages:
- **减少分布偏移**：当存储的情景片段与训练语料中的典型事件跨度匹配时，回忆提示更接近预训练分布，提高了 token 预测概率  
  **Reduced Distributional Shift**: When stored episodes match typical event spans found in training corpora, recall prompts resemble the pre-training distribution, improving token prediction probabilities
- **增强检索精度**：存储「人类尺度」事件的记忆索引操作的是语义纠缠较少的单元，提高了检索中的信噪比  
  **Enhanced Retrieval Precision**: Memory indices storing "human-scale" events operate on semantically less entangled units, increasing signal-to-noise ratio in retrieval

## 技术实现方法  
## Technical Implementation

### 数据预处理  
### Data Preprocessing

由于我们的生产系统增量处理原始情景数据，我们重用了主题分割策略。这体现了情景记忆创建的核心哲学："与人类记忆事件情景的粒度保持一致"。虽然我们的方法可能看起来低效且简单，但这反映了我们 MVP 为了清晰展示方法而做出的简化。在生产环境中，我们采用更具成本效益和高效的方法。  
Since our production system processes raw episodic data incrementally, we reused our topic segmentation strategy. This embodies the core philosophy of episodic memory creation: "aligning with the granularity of human memory event episodes." While our approach may appear inefficient and simplistic, this reflects the simplifications made for our MVP. In production, we employ more cost-effective and efficient methods.

对于情景生成，我们选择了最直接的版本来最好地说明我们的方法，仅使用 gpt-4o-mini/gpt-4.1-mini 进行情景记忆提取。  
For episode generation, we chose the most straightforward version that best illustrates our approach, using only gpt-4o-mini/gpt-4.1-mini for episodic memory extraction.

### 检索策略  
### Retrieval Strategy

我们为每个用户的情景记忆建立了最小的 BM25 索引。这可能会引起疑问，但这同样是一种简化。我们的生产系统采用结合稀疏（BM25）和密集（向量检索）方法的混合检索策略，以平衡召回和语义匹配能力，并针对特定业务需求定制不同的重排策略。  
We established a minimal BM25 index for each user's episodic memories. This might raise questions, but again, it's a simplification. Our production system employs a hybrid retrieval strategy combining sparse (BM25) and dense (vector retrieval) methods to balance recall and semantic matching capabilities, with different reranking strategies tailored to specific business needs.

在预处理完成后，后续过程相对简单。我们检索前 20 个结果，让 gpt-4o-mini/gpt-4.1-mini 生成响应，并遵循与其他项目几乎相同的评估方法。  
With the preprocessing complete, the subsequent process is relatively straightforward. We retrieve the top 20 results, have gpt-4o-mini/gpt-4.1-mini generate responses, and follow an evaluation approach nearly identical to other projects.

## 未来路线图  
## Future Roadmap

1. [计划开源] 添加「语义记忆」能力，用于改善情景记忆丢失原文中名称、地点等信息的问题。  
   [Planned open source] Add "semantic memory" capability to address the issue of episodic memory losing information such as names and locations from the original text.

2. 仅仅拥有特定事件的情景记忆是不够的。我们希望通过相似性度量等方法自动聚合情景片段，形成更长期和通用的高级情景片段。  
   Having episodic memory of specific events alone is insufficient. We hope to aggregate episodes through similarity measures and other methods to form longer-term and more general high-level episodes.

## FAQ

### 1. Nemori 在 LoCoMo 数据集的分数看起来跟 MemOS 提升并不大，你们的记忆方案有什么显著的优势吗？  
**Q: Nemori's score on the LoCoMo dataset doesn't seem much higher than MemOS. What are the significant advantages of your memory approach?**

MemOS 是一个非常优秀的方案，并且在我们真实的系统中，有非常多类似模块设计，在实际的落地场景中，记忆还是一个与业务强相关的问题，因此我们并无意对比框架本身的优劣。我们想表达的是，沿着「与人类记忆事件的颗粒度对齐」这一个简单而深刻的洞察作为起点，**在特定类型的任务中**只需要很简单的方法就能媲美复杂的记忆框架。在这个版本中，甚至都没有使用 Embedding 来增强语意相关性的召回效果。因此，如果只是想提升这个数据集的分数，还有非常大的提升空间。大家感兴趣可以自行尝试引入一些常规的优化方案，比如情景切割后的边界上下文修复，策略性的带入部分原文，引入混合召回+Rerank，或者进行记忆关联融合等，这些都是我们在真实系统中使用并且被证明有效的方法。  
MemOS is an excellent solution, and in our real-world systems, we have many similar module designs. In practical scenarios, memory is still a business-specific problem, so we do not intend to compare the merits of the frameworks themselves. What we want to express is that starting from the simple yet profound insight of "aligning with the granularity of human memory events," **for certain types of tasks**, even simple methods can rival complex memory frameworks. In this version, we didn't even use embeddings to enhance semantic recall. Therefore, if you just want to improve the score on this dataset, there is still a lot of room for improvement. You are welcome to try common optimization strategies, such as boundary context repair after episode segmentation, strategically including parts of the original text, introducing hybrid retrieval + rerank, or memory association fusion. These are all methods we use and have proven effective in real systems.

### 2. 为什么检索之后只用情景，不考虑带上原文？  
**Q: Why do you only use episodes after retrieval, without including the original text?**

因为我们真实的系统中，有另一套方案来解决「某些信息仅能在原文获取」的问题（后续计划开源），大概的方式就是会有选择性的 fusion 关键性语义记忆回情景记忆，在这里我们简化掉了。同时我们不觉得直接带全部原文或者带 top-x 个原文，会有特别明显的提升。直觉上个别 case 可能可以答对，但是整体上变化不会太大（这里主要受限于 gpt-4o-mini），感兴趣的可以试一下。  
In our real system, we have another solution (planned to be open-sourced) to address the issue that "some information can only be obtained from the original text." The general approach is to selectively fuse key semantic memories back into episodic memory, which we have omitted here for simplicity. We also don't think that directly including all or the top-x original texts would bring significant improvement. Intuitively, it may help in some cases, but overall, the change won't be substantial (mainly limited by gpt-4o-mini). Feel free to try it out if you're interested.

### 3. Nemori 的实验数据中，每个问题消耗的 token 明显超过了其他方法，所以这个效果是靠大量的上下文换来的吗？  
**Q: In Nemori's experimental data, each question consumes significantly more tokens than other methods. Is this effect achieved by using a large amount of context?**

情景记忆的构建策略，会让情景表述比一般的总结性语句长，在同样取 topk = 20 的条件下，总 token 数要比其他方法高不少，但根据我们的经验，即使 topk = 10（即 token ≈ 减半），可能表现差别不会太大。反过来，如果在一个合理范围内提升窗口上下文的使用对性能有提升，为什么不呢？  
The construction strategy of episodic memory makes episode descriptions longer than typical summary statements. With the same topk = 20, the total token count is much higher than other methods, but in our experience, even with topk = 10 (i.e., about half the tokens), the performance difference may not be significant. Conversely, if increasing the context window within a reasonable range improves performance, why not do it?

### 4. 该方法中关于情景的表述，有大量的类似「the previous Friday (June 23, 2023)」这样的表述，是面向评估集的定向优化吗？  
**Q: In your method, there are many expressions like "the previous Friday (June 23, 2023)" in the episodes. Is this a targeted optimization for the evaluation set?**

在我们团队真实的 Agent 系统中，有专门的时间增强处理过程，并将结果挂载在情景的元数据中，帮助在不同场景的业务中准确找到相对/绝对时间。我们在 MVP 实现做了简化，直接使用这样的形式拼接在情景正文中。而如果这一个操作如果对 LoCoMo 数据集有帮助（其实我并不太确定，因为第一版就是这么做的），那么说明这个数据集的构造方式，与我们真实的业务场景比较接近，也是非常符合预期的。  
In our team's real Agent system, there is a dedicated time enhancement process, and the results are attached to the episode metadata to help accurately locate relative/absolute time in different business scenarios. In the MVP, we simplified this by directly concatenating such expressions in the episode text. If this operation helps with the LoCoMo dataset (I'm not entirely sure, as this is how we did it in the first version), it means the dataset's construction is quite close to our real business scenarios, which is very much in line with expectations.

### 5. 在代码中，有很多看起来不知道用在哪里的处理，比如 EpisodeLevel、Time Gap 的计算等，为什么在一个 MVP 项目里面会设计这些元素？  
**Q: There are many elements in the code, such as EpisodeLevel and Time Gap calculation, that don't seem to be used. Why design these in an MVP project?**

50% 是生产项目迁移时的一些功能没被仔细剥离，50% 是 Claude Code 自由发挥。  
50% are features not carefully stripped out during production project migration, and 50% are Claude Code's creative freedom.

### 6. 情景记忆这个方法有在其他场景证明有效性吗？  
**Q: Has the episodic memory method proven effective in other scenarios?**

我们的本意，就是在「AI Agent 作为用户社交/办公的助理」和「通用 ChatBot」两个场景下，设计一种更高效的「记忆索引方式」。所以虽然我们没做过太多实验，但是基本可以推断出来，这个方法拿去做文档场景、知识库场景的记忆，应该是不会有什么直接提升的。  
Our intention is to design a more efficient "memory indexing method" for two scenarios: "AI Agent as a user's social/office assistant" and "general ChatBot." So although we haven't done many experiments, it can be basically inferred that using this method for document or knowledge base memory scenarios is unlikely to bring direct improvements.

## 特别感谢  
## Special Thanks

MemOS 团队——我们从他们的项目中分叉并扩展了评估框架以支持 Nemori 基准测试。  
MemOS team—we forked their project and extended the evaluation framework to support Nemori benchmarking.

**Nemori** - 赋予 AI 智能体长期记忆以驱动其自我进化 🚀  
**Nemori** - Endowing AI agents with human-like episodic memory to drive their evolution 🚀
