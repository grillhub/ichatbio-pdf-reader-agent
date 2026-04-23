from pathlib import Path

import dotenv

# Default load_dotenv() only checks the process cwd; pytest may use another cwd.
# Repo-root .env matches how other ichatbio-*-agent projects are configured.
_REPO_ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(_REPO_ROOT / ".env")
dotenv.load_dotenv()