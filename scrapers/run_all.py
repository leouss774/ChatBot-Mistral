"""
Lance les 3 scrapers de l'étape 2 dans l'ordre : site, GitHub, papiers.

Usage:
    python scrapers/run_all.py
"""

import subprocess
import sys
import os

SCRAPERS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["site_scraper.py", "github_scraper.py", "arxiv_scraper.py"]


def main():
    for script in SCRIPTS:
        print(f"\n=== {script} ===")
        result = subprocess.run([sys.executable, os.path.join(SCRAPERS_DIR, script)])
        if result.returncode != 0:
            print(f"Attention: {script} a rencontré une erreur (code {result.returncode})")


if __name__ == "__main__":
    main()