# ichatbio-pdf-reader-agent

PDF reader agent of iChatBio.

## Quickstart

*Requires python 3.10 or higher*

Set up your development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

Run the server:

```bash
uvicorn --app-dir src agent:create_app --factory --reload --host "0.0.0.0" --port 9999
```

You can also run the agent server as a Docker container:

```bash
docker compose up --build
```

If everything worked, you should be able to find your agent card at http://localhost:9999/.well-known/agent.json.
