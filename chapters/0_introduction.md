# Build Your Own Vector Database

Do you wonder how a "generative search" feature like Google's works?
How Spotify finds similar music to recommend?
How you can search images on the web with just a description?
All of these applications are built on top of **vector databases**.

Vector databases have rapidly grown in popularity recently, thanks to concepts like retrieval augmented generation (RAG).
This has led to a flurry of start-ups in the space, like Pinecone, Weaviate, etc., which come on top of an already rich and thriving open source ecosystem.

As vector databases grow in popularity, I think it's worthwhile to take a step back and understand how the tools we're employing work.
The goal of this tutorial is to help you understand what vector databases are, why they're important, and how they work.

In particular, we will walk through an hierarchical navigable small worlds (HNSW), a new(-ish) approach to fast approximate nearest neighbor search.
After spending a few evenings going through this, you will have a deep grasp of HNSW and **a fully working vector database of your own**.

```
⚠️ Much like cryptography, it's probably best that you stick with current, known implementations (e.g. FAISS, hnswlib, uSearch, etc.) rather than rolling your own for production.
```

## What this tutorial is

This tutorial will walk you through the process of implementing a simple vector database, and then making it more efficient by understanding an implementing a new, and popular, algorithm called hierarchical navigable small worlds (HNSW).

## What this tutorial isn't

Although I will try to cover the introductory concepts at a high-level, this tutorial probably isn't the best introduction to concepts like: "what is a vector".

Similarly, I will not go into advanced optimization, serialization, or memory management strategies.
Once you've finished building `tinyhnsw` you'll have a small, working, and reasonably efficient vector database.
However, you should see this as a launching off point more than anything!
You could make it more efficient, or you could add different approximate nearest neighbors algorithm, or seek to optimize the serialization and space requirements, or make it resistant to crashes, or so many other things!
The neat thing about small libraries that you can fully understand is that they're wonderful experimental test-beds.

## Some Concepts

- `vector database` - a vector database is a piece of software that makes it easy to store and search vectors.
- `retrieval augmented generation (RAG)` - retrieval augmented generation is an approach to using generative language models where relevant context is retrieved from a larger source (e.g., a document, or the web) and given to the model. Originally, it was trained end-to-end, but now it mostly means feeding the context into the prompt. It's commonly cited as a way of reducing hallucinations.
- `nearest neighbors search` - nearest neighbors is a way of searching a vector database: given a query vector, find the closest stored vectors to that. For instance, we may choose to encode a text query as a vector, and use it to search a stored collection of image vectors.
- `approximate nearest neighbors (ANN)` - nearest neighbors search can be an expensive operation. For instance, if you have 100 million vectors stored, every time you want to search them you have to do 100 million comparisons. In order to make this more computationally tractable, approximate nearest neighbors (ANN) search makes some simplifications at the trade-off of accuracy.
- `hierarchical navigable small worlds (HNSW)` - hierarchical navigable small worlds is a nearest neighbor search algorithm developed in 2016 that uses a hierarchical collection of graphs to do a time-efficient approximate nearest neighbors search.
