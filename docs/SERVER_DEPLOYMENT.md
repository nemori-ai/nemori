# Nemori evaluation server deployment

The dashboard is designed to stay private on the server and be opened through
an SSH tunnel. Do not expose port `18765`, PostgreSQL, Qdrant, or Ollama directly
to the LAN or Internet.

## Server runtime

- Dashboard: `127.0.0.1:18765`
- PostgreSQL 16 (Debian Bookworm): `127.0.0.1:15432`
- Qdrant HTTP / gRPC: `127.0.0.1:16333` / `127.0.0.1:16334`
- Ollama: `127.0.0.1:21434`
- Docker bridge: `10.231.0.0/24` by default; override with
  `NEMORI_DOCKER_SUBNET` if the server already routes that subnet
- Python and isolated SWE tools: managed by `uv`
- SWE execution: native Docker on Linux; WSL only on Windows

The website provides two deliberately separated arms:

- `API + SWE`: answer-free task plus the Coding Agent API, with no Nemori
  memory in the input.
- `Nemori + SWE`: the same task and Agent, with native Nemori repository
  memory built and retrieved before the Agent runs.

## Open the server website from Windows

Run:

```powershell
./scripts/open-server-dashboard.ps1
```

Then open <http://127.0.0.1:18765>. Keep the PowerShell window open while using
the site. The script relies on normal SSH authentication and never stores a
password.

Equivalent manual command:

```powershell
ssh -N -L 18765:127.0.0.1:18765 liangshengyu@192.168.106.20
```

## User-level persistence

On servers where the deployment account has no `sudo` access, run
`scripts/run-server-dashboard.sh` with `nohup` and register that same command
as a user `@reboot` crontab entry. Runtime logs are stored in the project and
the process only binds to `127.0.0.1:18765`. The three Compose services use
`restart: unless-stopped`, so Docker restores them after a server reboot.

## Disk-safe SWE defaults

For a host with less than 200 GB free, start with one task, one Harness worker,
`base` cache, and cleanup enabled. `instance` cache can consume multiple
terabytes and should not be selected on a shared server.

The runtime budgets and cache policy can be changed on the website under
**Environment & Connections > SWE Runtime**. Saved settings apply to the next
run.
