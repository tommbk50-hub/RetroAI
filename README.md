# RetroAI

RetroAI combines AiZynthFinder route search, RDKit analytics, Open Reaction Database precedents, and optional Gemini enrichment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tommbk50-hub/RetroAI/blob/main/Retrosynthesis_Agent.ipynb)

Run both notebook cells to install the package, download the USPTO models and ZINC stock, and launch the Ketcher workbench. Set `KDD_GEMINI_API_KEY` in Colab Secrets for Gemini features. For headless use, set it as an environment variable instead. `HF_TOKEN` is optional.

## Local development

From the repository root with Python 3.11 or newer:

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Non-UI modules and tests do not require Colab, model data, or API keys. For route searches, use the notebook's separate AiZynthFinder and Reaction-Utils install and run `download_public_data data` from the directory in which you will run the search.

The notebook installs Reaction-Utils with the engine using `--no-deps` because its published metadata would otherwise downgrade `urllib3` to a vulnerable 1.x release. RetroAI declares the ONNX engine's required libraries itself. The package also permits protobuf 5.29.6 because published `ord-schema` wheels require protobuf below 6; pinning protobuf to 6.31.1 or newer would make the base install unsatisfiable.


## Live Repository Map
<!-- AUTO-MAP:START -->
<!-- AUTO-MAP:END -->
