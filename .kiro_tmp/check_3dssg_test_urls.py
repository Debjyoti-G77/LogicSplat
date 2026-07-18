"""
Check for test-split specific files in the 3DSSG download server.
"""
import urllib.request

# Try various URL patterns for test annotations
urls = [
    # Objects files
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/objects.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json",
    # Split-specific files
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships_train.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships_val.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset/relationships_test.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships_train.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships_val.json",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG/relationships_test.json",
    # Maybe a zip or tar
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG.zip",
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3DSSG_subset.zip",
    # Check 3RScan paths
    "https://www.campar.in.tum.de/public_datasets/3RScan/3RScan.json",
    # Maybe the full 3RScan.json has more info
    "https://www.campar.in.tum.de/public_datasets/3DSSG/3RScan.json",
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
