# RetroAI

RetroAI combines AiZynthFinder route search, RDKit analytics, Open Reaction Database precedents, and optional Gemini enrichment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tommbk50-hub/RetroAI/blob/main/Retrosynthesis_Agent.ipynb)

<img width="1103" height="625" alt="image" src="https://github.com/user-attachments/assets/6529d147-a8db-49fc-9b84-ecef22a24232" />


Run both notebook cells to install the package, download the USPTO models and ZINC stock, and launch the Ketcher workbench. Set `KDD_GEMINI_API_KEY` in Colab Secrets for Gemini features. For headless use, set it as an environment variable instead. `HF_TOKEN` is optional.

## How it works?
Retrosynthesis is planning backward: start with the molecule you want to make and look for simpler starting materials that could lead to it. RetroAI suggests several possible routes so you can compare them.
### Plan a route
1. **Choose a target.** Draw it in the Ketcher sketchpad, or enter a name, database ID, or SMILES (a text representation of a molecule) in the search box and click **Load**. You can also pick a built-in example.
2. **Run the search.** In **Synthesis**, click **Plan Synthesis Route**. The defaults are enough for a first run. In the search settings, you can limit the number of steps or how long the search runs. Use **Lock Selected** to keep only routes that contain your selected fragment in an intermediate or starting material, or **Register Lab Stock** to count materials you already have.
3. **Read the route cards.** **Solved** means all starting materials are listed in stock; **Partial** means some are missing. Open **Why might this fail?** to see flagged risks, and inspect individual steps for similar recorded reactions where available.
### How routes are found and scored
AiZynthFinder uses patterns learned from patent reactions to suggest ways to break the target into simpler pieces. It explores different branches in this way, looking for paths that lead back to starting materials listed in the ZINC stock catalog. RetroAI also checks the proposed routes against any lab stock you registered.
RetroAI looks for similar experiments in the **Open Reaction Database (ORD)**. It uses evidence such as reported yields and failures, when available, to score the support for each step and flag the least-supported step as the **bottleneck**. The ORD confidence score summarizes that evidence; it is not a probability that the route will work.
ORD records are standardized and desalted before indexing. Validation uses a local DuckDB funnel: exact product-connectivity hits and DRFP candidates are shortlisted first, then chemically checked with reaction-center SMARTS and functional-group changes before empirical ranking. Missing precedents are explicitly marked **Unvalidated** rather than receiving synthetic yield evidence; ELN scaling then requires an explicit scenario yield until empirical data is available.
The overall ranking also considers the search score, the number of steps, starting material availability, and estimated safety. Use **Rank By** to emphasize practical considerations such as stock and safety (**CRO-Practical**), shorter routes (**Discovery (Fast)**), stronger experimental evidence (**Evidence-Backed (ORD)**), or estimated safety and environmental impact (**Green / Safe**). **Re-classify & Re-score (ORD + AI)** refreshes the evidence and scores for existing routes without starting a new search. A high score is not a substitute for lab validation or safety review.
### Explore the other tabs
- **Reaction** checks a reaction you draw or enter against similar recorded experiments.
- **Analytical** predicts signals you might see in a mass spectrometry report. With Gemini enabled, it can also help interpret a pasted report.
- **Chemist AI Agent** and **Structure Q&A** use Gemini to discuss your routes, molecules, and supporting evidence. Route search itself does not require Gemini.

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
