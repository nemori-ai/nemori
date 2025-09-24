"""Minimal example using NemoriMemory facade."""

from datetime import datetime

from nemori import NemoriMemory


def main() -> None:
    with NemoriMemory.from_env() as memory:
        user_id = "example-user"
        memory.add_messages(
            user_id,
            [
                {
                    "role": "user",
                    "content": "I love testing Nemori!",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "role": "assistant",
                    "content": "Noted. I'll remember that you enjoy testing.",
                    "timestamp": datetime.now().isoformat(),
                },
            ],
        )
        memory.flush(user_id)
        memory.wait_for_semantic(user_id)

        results = memory.search(user_id, "testing", search_method="vector")
        print("Search results:", results)


if __name__ == "__main__":  # pragma: no cover
    main()
