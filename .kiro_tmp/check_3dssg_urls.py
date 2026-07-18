"""
Check what's available at the 3DSSG download URLs.
The preparation.sh downloads from 3DSSG_subset. Let's check if there's a full version.
"""
import urllib.request
import json

# URLs to check
urls = [
    # The subset version (what preparation.sh downloads)
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships.json",
    # Possible full version URLs
    "https://www.campar.in.tum.de/public_datasets/3DSSG/relationships.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_full/relationships.json",
    # Check if there's a relationships_test.json
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships_test.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/relationships_test.json",
    # Check objects
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/objects.json",
]

for url in urls:
    try:
        req = urllib.request.Request(url, method='HEAD')
        resp = urllib.request.urlopen(req, timeout=10)
        size = resp.headers.get('Content-Length', 'unknown')
        print(f"  EXISTS ({size} bytes): {url}")
    except urllib.error.HTTPError as e:
        print(f"  {e.code}: {url}")
    except Exception as e:
        print(f"  ERROR ({type(e).__name__}): {url}")
