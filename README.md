# Facility Location

Where to open warehouses — uncapacitated facility location

**Live demo:** https://facility-location.griffith-pse.com  
**Home:** https://griffith-pse.com

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

## Deployment

Auto-deploys to Fly.io on every push to `main` via
`.github/workflows/deploy.yml`. The `Dockerfile` builds a Python 3.12 image
and installs everything from `requirements.txt`; `fly.toml` configures
auto-stop machines. Custom domain wired through Cloudflare DNS.

- **Machine**: `shared-cpu-1x` · 1 GB RAM · single region (`ord`) · `min_machines_running=0` (auto-stops on idle).
- **Cost ceiling**: ~$3.89/mo if traffic kept the VM awake 24/7. Realistic on idle-heavy demo traffic: well under $1/mo per app. Bandwidth is effectively free under Fly's 100 GB/mo egress allowance.

## Files

- `app.py` — Streamlit UI and computation
- `requirements.txt` — Python deps
- `favicon.png` — Griffith PSE blackletter G favicon
- `Dockerfile`, `fly.toml`, `.dockerignore` — Fly.io production image config
- `.github/workflows/deploy.yml` — auto-deploy pipeline

## References

[1] A. A. Kuehn and M. J. Hamburger, "A Heuristic Program for Locating
Warehouses," *Management Science*, vol. 9, no. 4, pp. 643–666, 1963.
[INFORMS](https://pubsonline.informs.org/doi/10.1287/mnsc.9.4.643)

[2] D. Erlenkotter, "A Dual-Based Procedure for Uncapacitated Facility
Location," *Operations Research*, vol. 26, no. 6, pp. 992–1009, 1978.
[INFORMS](https://pubsonline.informs.org/doi/10.1287/opre.26.6.992)

[3] G. Cornuéjols, G. L. Nemhauser, and L. A. Wolsey, "The Uncapacitated
Facility Location Problem," in *Discrete Location Theory*,
P. B. Mirchandani and R. L. Francis, Eds. New York: Wiley, 1990,
pp. 119–171.

[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird, B. L. Nicholson,
J. D. Siirola, J.-P. Watson, and D. L. Woodruff, *Pyomo — Optimization
Modeling in Python*, 3rd ed. Cham: Springer, 2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
