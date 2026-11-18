# CPU development container

Build and run from the repository root. The image uses Python 3.11.14, the
PyTorch 2.9.1 CPU build, and the [CPU workflow’s dependency pins](../.github/requirements-cpu.txt).

```bash
docker build -f docker/Dockerfile -t vaidylang:cpu .
docker run --rm vaidylang:cpu
```

The default command runs the unit suite. For a quick parity check, override it.

```bash
docker run --rm vaidylang:cpu python harness.py verify --tiny \
  --prompt-lengths 1 8 32 128 --new-tokens 12
```

In VS Code with Docker running, install the Dev Containers extension and choose
**Dev Containers: Reopen in Container**. The [Dev Container configuration](../.devcontainer/devcontainer.json)
uses this image and opens the checked-out workspace as the `developer` user.

The plain container suite skips CUDA tests. The published-tokenizer unit test
also skips unless `VAIDYLANG_DEEPSEEK_METADATA` points to the pinned assets.
The [CPU workflow](../.github/workflows/cpu.yml) fetches those assets for its
upstream checks and builds and tests the container on x86-64 and ARM64.
