# ichatbio-pdf-reader-agent

PDF reader agent for iChatBio that extracts and processes PDF documents.

## Features

- **PDF download and processing** — Downloads PDFs from URLs and extracts text content
- **Structured output** — Returns extracted text as artifacts that iChatBio can use for answering questions

## Quickstart

*Requires python 3.12 or higher*

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

### Example

User message: "Can you read this PDF: https://arxiv.org/pdf/2408.09869"

The agent will:
- Detect the PDF URL
- Download and process the PDF
- Extract all text content
- Make it available for iChatBio to answer questions about the document

## Installation Notes

```bash
pip install pypdf
```

## License

This project uses the iChatBio SDK and follows the same licensing terms.
