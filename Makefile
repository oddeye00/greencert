.PHONY: env smoke audit figures paper public-repo

env:
	python scripts/check_reproduction_environment.py

smoke:
	python reproduce.py smoke

audit:
	python reproduce.py artifact-audit

figures:
	python scripts/reproduce_figures.py --check-determinism

paper:
	python scripts/build_arxiv_release.py

public-repo:
	python scripts/build_public_repository.py

