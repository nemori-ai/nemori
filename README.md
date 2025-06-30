# Nemori: Nature-Inspired Episodic Memory for AI

*Read this in other languages: [中文](README-zh.md)*

## Project Overview

Nemori-AI empowers AI with human-like episodic memory. Nemori stores experiences as natural, event-centric traces, enabling precise recall when it matters. Vision: every piece of data remembered and retrieved as intuitively as human recollection.

While previous systems like Mem0, Supermemory, and ZEP have made remarkable attempts at AI memory, achieving advanced performance on benchmarks such as LoCoMo and LongMemEval, Nemori introduces an innovative and minimalist approach centered on aligning with human episodic memory patterns.

## Experimental Results

To highlight the superiority of Nemori, we conducted evaluations on both LoCoMo and LongMemEval benchmarks, comparing against previous state-of-the-art approaches:

### LoCoMo Benchmark Results

On the LoCoMo (Long-Context Conversation Modeling) dataset, Nemori demonstrates exceptional performance:

![LoCoMo Benchmark Results](figures/results_on_locomo.png)

### LongMemEval-s Benchmark Results

On the LongMemEval-s dataset, Nemori also achieves leading performance:

![LongMemEval Benchmark Results](figures/results_on_longmemeval_purple.png)

## Design Philosophy

When we humans recall past events, our minds often flash with related images, actions, or sounds. Our brains help us remember by essentially making us re-experience what happened at that time - this memory mechanism is called episodic memory.

Nemori's design inspiration comes from human episodic memory. Nemori can autonomously reshape conversations between humans, between humans and AI agents, or between AI agents into episodes. Compared to raw conversations, episodes have more coherent causal relationships and temporal expression capabilities. More importantly, the expression of episodes aligns to some extent with the granularity of our human memory recall, meaning that as humans, we are likely to ask questions about episodes that are semantically closer to the episodes themselves rather than the original messages.

## Future Roadmap

1. Having episodic memory for specific events alone is insufficient. We hope to aggregate episodes through methods such as similarity measures to form more long-term and general high-level episodes.

2. We designed Nemori from an anthropomorphic perspective. We are still uncertain whether future AI assistants' memory mechanisms will be fundamentally different from humans. We will conduct deeper thinking on this aspect.

This repository represents only a minimalist version of our work, but you can refer to the documentation in the evaluation folder to see our benchmark scores.

**Nemori** - Endowing AI agents with long-term memory to drive their self-evolution 🚀

