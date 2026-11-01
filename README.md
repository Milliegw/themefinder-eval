# ThemeFinder evaluation

An evaluation of i.AI's open-source [ThemeFinder](https://github.com/i-dot-ai/themefinder),
the theme-finding pipeline behind the Consult tool for analysing public consultation responses.

## Question

How accurately does ThemeFinder map consultation responses to themes, and can an
LLM judge be trusted to score its output?

## Status

Work in progress.

- [x] Metric functions (precision, recall, F1) written by hand, tested, and checked against scikit-learn
- [x] Bootstrap confidence intervals
- [x] ThemeFinder running locally
- [ ] Hand-labelled golden set of consultation responses
- [ ] Score ThemeFinder's theme mapping against the golden set
- [ ] LLM-as-a-judge with DeepEval, checked against human labels (Cohen's kappa)
- [ ] Same evaluation in Inspect
- [ ] Compare two models with a paired test

## Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python -m venv .venv               # create (once per project)
source .venv/bin/activate          # switch it on (every new terminal)
pip install pandas                 # install into this project only
pip freeze > requirements.txt      # record what's installed
deactivate                         # switch it off
python run_themefinder.py


ThemeFinder runs locally through Ollama's OpenAI-compatible endpoint (`ollama pull llama3.1`).

## Findings so far

**Small test sets give unreliable scores.** On 8 examples, F1 = 0.75 had a 95% bootstrap
interval of 0.0–1.0, so the score alone could not support any decision.

**Setup issues with ThemeFinder 0.8.2:**
- The PyPI usage example passes a LangChain model, but 0.8.2 expects its own `LLM` protocol
  (`themefinder.llm.OpenAILLM`). Passing a LangChain Ollama model fails with
  `unexpected keyword argument 'output_model'`.
- ThemeFinder fails with pandas 3 (`astype(int)` on a string column in
  `llm_batch_processor.py`). Pinned `pandas<3`.

## Next steps

Build a golden set of hand-labelled responses from a public consultation and evaluate
ThemeFinder's theme mapping against it.
