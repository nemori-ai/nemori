from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "dashboard.app:app",
        host=os.getenv("NEMORI_DASHBOARD_HOST", "127.0.0.1"),
        port=int(os.getenv("NEMORI_DASHBOARD_PORT", "8765")),
        reload=False,
        access_log=False,
    )


if __name__ == "__main__":
    main()
