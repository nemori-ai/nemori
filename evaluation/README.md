# Evaluation Guidance
Special thanks to the MemOS team. We forked their project and extended the evaluation framework to support Nemori benchmarking.

## clone memos for evaluation
```sh
cd evaluation
git clone https://github.com/MemTensor/MemOS memos
cd memos
git checkout v0.2.0
```

## apply patch
```sh
git apply --whitespace=nowarn ../nemori-eval-patch-for-memos-v0.2.0.patch
```

## create your python env and install poetry(chosen by memos)
...

## install eval requirements
> If poetry reports "project" not found in pyproject.toml, simply add a [project] section as needed.
```sh
poetry install --with eval
```

## install nemori
```sh
poetry add ../../
```

## setup
```sh
export OPENAI_API_KEY=xx
export CHAT_MODEL=gpt-4o-mini # gpt-4.1-mini
export EVAL_VERSION="nemori-eval"
```

## run eval
```sh
cd evaluation
python scripts/locomo/locomo_ingestion.py --lib nemori --version $EVAL_VERSION --workers 10
python scripts/locomo/locomo_search.py --lib nemori --workers 10 --version $EVAL_VERSION
python scripts/locomo/locomo_responses.py --lib nemori --version $EVAL_VERSION
python scripts/locomo/locomo_eval.py --lib nemori --workers 10 --version $EVAL_VERSION
python scripts/locomo/locomo_metric.py --lib nemori --version $EVAL_VERSION
```
