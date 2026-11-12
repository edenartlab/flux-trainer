"""Reproduce the Python environment of the last Vertex AI image (March 2025).

That image was built from this repository's Dockerfile, which pinned only some
packages; the rest were whatever was newest on the day. This app replays the
Dockerfile's three install steps, in the same order, with the same interpreter
and pip (Ubuntu 22.04's Python 3.10 and pip 22.0.2), against the package index
as it was on CUTOFF, and prints `pip freeze` after each step. The last block of
the output is committed as `requirements.lock`, which the Dockerfile installs.

    modal run modal_resolve_env.py > freeze.txt

Nothing is deployed and no GPU is used.

The installs run in `setup_dockerfile_commands`, before Modal adds its own
Python (3.11, under /usr/local) to the image, so Modal's packages and its newer
pip take no part in the resolution.
"""

import modal

# The day after the last code change that the production image contained.
CUTOFF = "2025-03-21"
KOHYA_COMMIT = "75933d70a15ac423f396c9d3df04d0d35fae6130"

APT = (
    "git wget curl build-essential libgl1-mesa-glx libgstreamer1.0-0 "
    "libgstreamer-plugins-base1.0-dev libglib2.0-0 python3 python3-pip "
    "python3-dev libmagic1 zip"
)

PIP = "/usr/bin/python3 -m pip"

# Serves PyPI as of CUTOFF on localhost:5000 for the duration of one build step.
TIME_MACHINE = (
    f"(/root/.local/bin/uvx --python 3.12 pypi-timemachine {CUTOFF} --port 5000 "
    ">/tmp/timemachine.log 2>&1 &) && "
    "for i in $(seq 180); do (echo > /dev/tcp/127.0.0.1/5000) 2>/dev/null && break; sleep 1; done"
)
INDEX = "--index-url http://127.0.0.1:5000/"

image = modal.Image.from_registry(
    "nvidia/cuda:11.8.0-runtime-ubuntu22.04",
    add_python="3.11",
    setup_dockerfile_commands=[
        "SHELL [\"/bin/bash\", \"-c\"]",
        "ENV DEBIAN_FRONTEND=noninteractive",
        f"RUN apt-get update && apt-get install -y --no-install-recommends {APT} "
        "&& rm -rf /var/lib/apt/lists/*",
        "RUN ln -s /usr/bin/python3 /usr/bin/python",
        "RUN curl -LsSf https://astral.sh/uv/install.sh | sh",
        f"RUN {PIP} freeze > /freeze-0-base.txt; {PIP} --version > /pip-version.txt",
        # Dockerfile step 1
        f"RUN {TIME_MACHINE} && {PIP} install {INDEX} timm==1.0.9 triton==3.1.0 "
        "bitsandbytes==0.44.0 requests==2.32.3 tqdm==4.67.1 pymongo==4.11.1 "
        "mediapipe==0.10.21 huggingface_hub==0.28.1 python-dotenv==1.0.1 "
        "boto3==1.36.21 python-magic==0.4.27 openai==1.61.0 toml==0.10.2 "
        f"&& {PIP} freeze > /freeze-1.txt",
        # Dockerfile step 2. The PyTorch index carries no upload dates; the three
        # caps exclude exactly the releases published after CUTOFF (2.7 / 0.22).
        "RUN printf 'torch<2.7\\ntorchvision<0.22\\ntorchaudio<2.7\\n' > /tmp/caps.txt && "
        f"{PIP} install torch torchvision torchaudio -c /tmp/caps.txt "
        "--index-url https://download.pytorch.org/whl/cu118 "
        f"&& {PIP} freeze > /freeze-2.txt",
        # Dockerfile step 3
        "RUN mkdir -p /app/flux-trainer && cd /app/flux-trainer && "
        "git clone https://github.com/kohya-ss/sd-scripts.git && cd sd-scripts && "
        f"git checkout {KOHYA_COMMIT} && {TIME_MACHINE} && "
        f"{PIP} install {INDEX} --no-cache-dir -r requirements.txt "
        f"&& {PIP} freeze > /freeze-3.txt",
    ],
)

app = modal.App("flux-trainer-resolve-env", image=image)


@app.function(timeout=600)
def freeze() -> str:
    import subprocess

    out = ["# " + subprocess.run(["/usr/bin/python3", "-VV"], capture_output=True, text=True).stdout.strip()]
    out.append("# " + open("/pip-version.txt").read().strip())
    for name in ("0-base", "1", "2", "3"):
        out.append(f"##### after step {name}")
        out.append(open(f"/freeze-{name}.txt").read())
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(freeze.remote())
