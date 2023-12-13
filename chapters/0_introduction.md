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

1. 

## What this tutorial isn't

Although I will try to cover the introductory concepts at a high-level, this tutorial probably isn't the best introduction to concepts like: "what is a vector".

## Some Concepts

- `vector database`
- `retrieval augmented generation (RAG)`
- `nearest neighbors`
- `approximate nearest neighbors (ANN)`
- `hierarchical navigable small worlds (HNSW)`
