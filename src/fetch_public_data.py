import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


URL = 'https://dataverse.harvard.edu/api/access/datafile/1374917'
EXPECTED_MD5 = 'a73e97aa143a3e21db3bf0cdfba68775'


def fetch(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        digest = hashlib.file_digest(destination.open('rb'), 'md5').hexdigest()
        if digest != EXPECTED_MD5:
            raise ValueError(f'Existing archive has unexpected MD5: {digest}')
        print(f'Archive already verified: {destination}')
        return

    digest = hashlib.md5()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            request = Request(URL, headers={'User-Agent': 'aircraft-rotation-selective-warning/1.0'})
            with urlopen(request, timeout=90) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
        actual = digest.hexdigest()
        if actual != EXPECTED_MD5:
            raise ValueError(f'Downloaded archive has unexpected MD5: {actual}')
        os.replace(temporary, destination)
        print(f'Downloaded and verified: {destination}')
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and verify the open 2008 airline on-time archive.')
    parser.add_argument('--output', type=Path, default=Path('data/2008.csv.bz2'))
    fetch(parser.parse_args().output)
