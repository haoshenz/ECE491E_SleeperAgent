"""Download the official CIFAR-10 archive with verified parallel byte ranges."""
import concurrent.futures
import hashlib
import os
from pathlib import Path
import time
import urllib.request

URL = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
MD5 = 'c58f30108f718f92721af3b95e74349a'


def main():
    root = Path('data')
    root.mkdir(exist_ok=True)
    dest = root/'cifar-10-python.tar.gz'
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == MD5:
        print('Official archive already verified')
        return
    head = urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30)
    size = int(head.headers['Content-Length'])
    chunk = 4*1024*1024
    parts = list(range(0, size, chunk))
    def fetch(start):
        end = min(start+chunk, size)-1
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    assert response.status == 206
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{size}'
                    data = response.read()
                    assert len(data) == end-start+1
                return start, data
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1)
    buffer = bytearray(size)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for done, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, p) for p in parts]), 1):
            start, data = future.result()
            buffer[start:start+len(data)] = data
            print(f'Downloaded {done}/{len(parts)} chunks', flush=True)
    assert hashlib.md5(buffer).hexdigest() == MD5, 'Official archive checksum mismatch'
    temp = dest.with_suffix('.download')
    temp.write_bytes(buffer)
    os.replace(temp, dest)
    print('Official CIFAR-10 archive verified', flush=True)


if __name__ == '__main__':
    main()
