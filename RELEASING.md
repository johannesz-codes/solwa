# Releasing SOLWA to PyPI

## One-time setup: PyPI Trusted Publishing

1. Log in to [pypi.org](https://pypi.org) (or create an account).
2. Go to **Your projects → Manage → Publishing** (or, for the first
   release, **Publishing** under *your account settings*).
3. Add a new **Trusted Publisher** with these values:

   | Field             | Value                              |
   |-------------------|------------------------------------|
   | GitHub owner      | `johannesz-codes`                  |
   | Repository        | `solwa`                            |
   | Workflow filename | `publish.yml`                      |
   | Environment       | *(leave empty)*                    |

4. Save. No API token is needed — GitHub Actions will authenticate
   via OIDC.

## How to make a release

1. **Bump the version** in both places:
   - `pyproject.toml` → `version = "X.Y.Z"`
   - `src/solwa/__init__.py` → `__version__ = "X.Y.Z"`

2. Commit and push the version bump:
   ```bash
   git add pyproject.toml src/solwa/__init__.py
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

3. **Create a GitHub Release**:
   - Go to *Releases → Draft a new release*.
   - Create a new tag, e.g. `vX.Y.Z`.
   - Set the title (e.g. `vX.Y.Z`) and add release notes.
   - Click **Publish release**.

4. The `publish.yml` workflow will build and upload to PyPI
   automatically.

## Local testing

```bash
# Editable install (for development)
pip install -e .

# Build the package locally
pip install build
python -m build

# Inspect the built distributions
ls dist/
```
