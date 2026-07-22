from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    uvicorn.run(
        "dashboard.app:app",
        host=os.getenv("NEMORI_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.getenv("NEMORI_DASHBOARD_PORT", "8765")),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
