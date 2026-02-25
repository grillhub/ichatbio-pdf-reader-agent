import uvicorn

import agent

if __name__ == "__main__":
    app = agent.create_app()
    uvicorn.run(app, host="0.0.0.0", port=9999)
