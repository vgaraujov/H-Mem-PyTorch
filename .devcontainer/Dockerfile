
FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | TERM=xterm sh

# Install Python 3.10
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive     && apt-get -y install python3.10 python3.10-venv python3-pip

# Create and activate virtual environment with Python 3.10
RUN python3.10 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Install IPython kernel and its dependencies
RUN pip install ipykernel &&     python -m ipykernel install --user --name python3 --display-name "Python 3.10.12"

# Note: requirements.txt will be installed after container creation
CMD [ -f "requirements.txt" ] && pip install -r requirements.txt || echo "No requirements.txt found"
