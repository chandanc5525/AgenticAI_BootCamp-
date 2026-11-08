# 02 · Advanced PyTorch

Part of the **Agentic AI BootCamp**. Six hands-on Jupyter notebooks that take you from PyTorch basics to the building blocks used in GenAI and agentic AI applications: transformers, fine-tuning with LoRA, RAG retrieval, tool routing and agent loops.

**Author:** Chandan C · **GitHub:** [github.com/chandanc5525](https://github.com/chandanc5525)

---


## Getting started

### Requirements

- Python 3.9+ (developed and tested on Python 3.12)
- PyTorch (developed and tested with 2.15)
- `numpy`, `matplotlib`, `jupyter` (or JupyterLab / VS Code)

```bash
git clone https://github.com/chandanc5525/AgenticAI_BootCamp-.git
cd AgenticAI_BootCamp-/02_advanced_pytorch

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install torch numpy matplotlib jupyter
jupyter notebook
```

Optional, only for the ONNX export cell in notebook 06:

```bash
pip install onnx onnxscript
```

### Hardware

Every notebook runs on a plain **CPU**. If CUDA or Apple MPS is available, the notebooks use it automatically. Notebook 04 trains a small GPT in about a minute on CPU.

## Design notes

- **Self-contained and offline.** The notebooks use synthetic or built-in data and small models, so no dataset downloads or API keys are needed.
- **Mechanics first.** Attention, LoRA, the vector store, the reranker and the agent loop are written from scratch so you can see how they work. Each notebook shows how to swap in production tools (Hugging Face, PEFT, sentence-transformers, Qdrant, LangGraph).
- **Demo scale.** The models are tiny, so results show behaviour, not production accuracy.
- **Tested.** All six notebooks were executed end to end on CPU. Two cells need optional extras: ONNX export (see above) and `torch.compile`, which needs a C++ compiler on CPU. Both are guarded and skip gracefully.

## Skills you will have after this module

- Read and write idiomatic PyTorch, including custom modules, data pipelines and training loops
- Explain and implement attention and a decoder-only transformer
- Fine-tune large models efficiently with LoRA, mixed precision, accumulation and checkpointing
- Build the retrieval, reranking and routing pieces of a RAG or agent system
- Evaluate retrieval quality and prepare models for serving

## What's next

Continue with the LangChain and LangGraph modules of the bootcamp, where the components built here are replaced by production frameworks.

---

*Maintained by [Chandan C](https://github.com/chandanc5525) · Agentic AI BootCamp*