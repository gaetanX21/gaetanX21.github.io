---
layout: post
title: "Adding salt to the Bitter Lesson"
date: 2025-05-04
description: "TL;DR: The \"Bitter Lesson\" of AI states that general methods that leverage computation are ultimately the most effective to build powerful AI systems. We propose to qualify this lesson by introducing the notion of signal-to-noise ratio (SNR) of the problem at hand. In domains such as quantitative finance and computational biology, I believe that the SNR is so low that Sutton's lesson may not directly apply."
tags: meta, learning
thumbnail: assets/img/posts/bitter_lesson/gpus_go_brrr.webp
---

In this post, I will briefly discuss Richard Sutton's *Bitter Lesson* of AI. I will also present a lesser-known counter-argument by Rodney Brooks, and finally I will add my own grain of salt to the discussion with a focus on the signal-to-noise ratio (SNR) of the problem at hand. I will illustrate this idea with two specific domains in which human priors have yet to be discarded: quantitative finance and computational biology.

---

<div class="row justify-content-center" id="fig-1">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/bitter_lesson/gpus_go_brrr.webp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. Sutton's Bitter Lesson, illustrated. <a href="https://horace.io/brrr_intro.html">Source</a>.
</div>

## I. Richard Sutton's Bitter Lesson

Sutton's *Bitter Lesson*[^sutton] begins with the following statement:
> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

Basically, Sutton's big idea is that trying to forcefully incorporate human knowledge in AI systems (which was essentially the norm before the deep learning revolution) hinders progress. Instead, leveraging vasts amounts of compute (and data) is the way to go. This implies that any attempt to inject human ingenuity into AI systems is doomed to fail, hence the "bitter" lesson. In his lesson, Sutton gives several good examples of this phenomenon, for instance in computer vision, where models using complicated human-designed features (e.g., SIFT) were quickly outperformed by deep learning methods that *learned* features directly from data.

In today's age of transformer models scaling up to trillions of parameters, Sutton's lesson seems more relevant than ever, and some veteran NLP researchers have certainly felt bitter seeing their carefully handcrafted models being outperformed by large language models (LLMs) trained on (somewhat random) internet text. Companies building LLMs certainly have reasons to believe in the bitter lesson. Rumor has it that memorizing Sutton's article is part of OpenAI engineers' work schedule[^openai]. Funnily enough, OpenAI itself got bitter-lessoned in 2024 when it created a fine-tuned version of its then-flagship `o1` model specifically for competitive programming, `o1-ioi`, which ended up being uniformly worse than the firm's next-generation general-purpose model, `o3`.


## II. Rodney Brooks' *Better* Lesson

A week after Sutton's post, Rodney Brooks published a blog post titled *A Better Lesson*[^brooks] in which he essentially argues that Sutton is wrong. As he carefully put it:
> "I think Sutton is wrong for a number of reasons."

Brooks lists a few reasons why he believes Sutton's lesson is wrong. His core thesis is that AI systems are still imbued with human knowledge, only now it is hidden in the choice of model architectures, and to a lesser extent in the curated datasets and the complex training pipelines. Besides, he argues that the current trend of scaling up models is not sustainable, notably because Moore's law is slowing down and frontierAI models' carbon footprint is becoming a cause for concern.


## III. My grain of salt: SNR matters

My (humble) view is that Sutton's Bitter Lesson is generally a good heuristic for AI research, but it should be taken with a grain of salt (!).
> "I believe that the signal-to-noise ratio (SNR) of the problem at hand matters a lot."

I will illustrate this idea on two specific domains in which human priors have yet to be discarded: quantitative finance and computational biology.

#### A. Quantitative Finance

Financial markets are notoriously noisy, as they are complex systems in which a myriad of heterogeneous agents interact with different objectives. As such, it's well-known that the SNR of financial data is extremely low. For that reason, robustness is a key concern in model selection and most market practitioners end up relying on the good ol' linear regression model, albeit augmented with a few hand-crafted biases. Although the industry is catching up with the latest AI trends (e.g. using LLMs for sentiment analysis), the SNR of financial data is so low that it is hard to imagine a future in which Sutton's lesson will be fully applicable. In fact, I would argue that **the SNR of financial data is so low that it is not even clear whether Sutton's lesson applies at all**. Can a 1B-parameter model trained on 1TB of (crappy) data outperform a 10-parameter linear model trained on 1MB of data? I don't know, but I wouldn't be surprised if it didn't!

#### B. Computational Biology

Data in computational biology is also very noisy, but for different reasons. Here I will focus on RNA-seq data[^rnaseq], which in a nutshell (within a nutshell) is tabular data of the form `n_cells x n_genes` where each row gives you for a given cell the expression level of the 20k or so (human) genes. As it stands, RNA-seq data has several issues, the most obvious one being that it is very sparse[^sparse] and hence difficult to work with. More importantly, RNA-seq data is "dirty" in the sense that it is collected in a wet lab by a human being (i.e. not a machine) who has their own way of doing things[^law-compbio]. This leads to what is called "batch effects", which are systematic differences between data collected in different experiments. In the context of NLP, this is like if I told you that the text data scraped on Wikipedia didn't follow the same distribution as the text data scraped on Reddit. That would certainly make matters difficult, right?

But there is a much deeper problem with RNA-seq data, which is that **gene expression fundamentally isn't a clean signal**, unlike text data in (curated) web corpuses. The key idea is that life as we know it is literally the result of a random process left unchecked for 4 billion years, in which the fittest pass their genes to the next generation. This explains why organisms are so monstrously complex (unlike computer systems, which are trivial in comparison), but also extremely robust. A good example of this is the notion of *biological pathways*, which can roughly be described as "a series of interactions of molecules in a cell that leads to a certain product or a change in the cell"[^pathway]. In computer systems, pathways are bijective: Function A triggers Function B, and that's it. In an organism, Gene A may trigger production of Protein B, but it may also trigger production of Protein C. And guess what, Gene C can also create Protein B under certain conditions. Oh and wait, the goal of creating Protein B was to produce a certain molecule, but it turns out that this molecule can also be produced by Gene D! And so on and so forth. In other words, biological pathways are not bijective, and this is a super important because **redundancy yields robustness**. For instance, if one pathway producing glucose in a cell breaks down for some reason, the cell can still produce glucose through other pathways that were created through random mutations, so it doesn't die! The most critical components of life, such as the immune system, have myriads of redundant pathways, which makes them extremely robust to perturbations. As such, the current attempts[^goldrush] to emulate the dazzling successes of transformer models in NLP by training large transformer architectures on RNA-seq data may ultimately prove futile, as the SNR of the data may simply be too low.


## Conclusion

The Bitter Lesson is a great heuristic for AI research, but it must be taken with a grain of salt. In particular, the SNR of the problem at hand matters a lot. In domains such as quantitative finance and computational biology, the SNR is so low that it is not even clear whether Sutton's lesson applies at all. In these domains, human biases and ingenuity are still critical to building effective AI systems.

---

**References**:

[^sutton]: Sutton, R. (2019). *The Bitter Lesson.* [Link](http://incompleteideas.net/IncIdeas/BitterLesson.html)
[^brooks]: Brooks, R. (2019). *The Better Lesson.* [Link](https://rodneybrooks.com/a-better-lesson/)
[^openai]: Medium (2024). *The Legendary OpenAI Engineer’s Must-Have Classic: A Bitter Lesson.* [Link](https://ai-engineering-trend.medium.com/the-legendary-openai-engineers-must-have-classic-a-bitter-lesson-1948e6ac6c4a)
[^rnaseq]: Wikipedia. (2023). *RNA-Seq.* [Link](https://en.wikipedia.org/wiki/RNA-Seq)
[^pathway]: Wikipedia. (2023). *Biological pathway.* [Link](https://en.wikipedia.org/wiki/Biological_pathway)
[^law-compbio]: A colleague of mine with deep expertise in the field quickly taught me that "the first rule of computational biology is that everyone does things their own way".
[^sparse]: Not only because most genes are not expressed in most cells, but also because genes with low expressions may not be captured during RNA sequencing.
[^goldrush]: Given the record amounts invested, one might even call it a *gold rush*.