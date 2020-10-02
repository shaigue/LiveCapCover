from pathlib import Path
import re

import requests

root = 'https://gvv-assets.mpi-inf.mpg.de/LiveCap/wp-content/uploads/2019/10/'
dst_root = Path(__file__).parents[2] / 'assets/original'
files = [
    'Skinning.blend',
    'init2.motion',
    'mohammad.calibration',
    'mohammad.mtl',
    'mohammad.obj',
    'mohammad.skeleton',
    'mohammad.skin',
    'segmentation.txt',
    'textureMap.png'
]
# add all of the background images
background_dir = 'background/'
for i in range(1, 31):
    files.append(background_dir + f'frame_{i:06d}.png')
# add the 2000 first images
images_dir = 'images/'
for i in range(1, 2001):
    files.append(images_dir + f'frame_{i:06d}.png')

# use instruction on https://gvv-assets.mpi-inf.mpg.de/LiveCap/?page_id=224 to get the command
# to get the correct wget_command


wget_command = """
wget --header="Host: gvv-assets.mpi-inf.mpg.de" --header="User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36" --header="Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9" --header="Accept-Language: he,en-GB;q=0.9,en;q=0.8" --header="Cookie: wordpress_logged_in_8e763b0bda6f5c16b7564ec7998639f6=234%7C1600604307%7CsRpqmHWaCawOXiC2ATSn4bEHSPnjQviMk4dkG8TxiVG%7C6f881ab94ae76e0bafcdd97c8669d0e230e6a833e93ee03468333eb224969a15; _ga=GA1.2.724769111.1598870114; CookieLawInfoConsent=eyJuZWNlc3NhcnkiOnRydWV9; viewed_cookie_policy=yes; cookielawinfo-checkbox-necessary=yes; wptp_terms_110=accepted" --header="Connection: keep-alive" "https://gvv-assets.mpi-inf.mpg.de/LiveCap/wp-content/uploads/2019/10/" -c -O 'Skinning.blend'
"""
# find all of the header properties
header = []
for m in re.finditer('--header="', wget_command):
    start = m.end()
    end = start
    mid = None
    for c in wget_command[start:]:
        if c == '"':
            break
        if mid is None and c == ':':
            mid = end
        end += 1
    header.append((wget_command[start:mid], wget_command[mid+2:end]))

print(header)
session = requests.Session()
for key, value in header:
    session.headers[key] = value

for file in files:
    print(f'getting file {file}...')
    response = session.get(root + file)
    if response.status_code != 200:
        print(f'request {response.request} has failed.')
        exit(-1)
    local_path = dst_root / file
    if not local_path.parent.is_dir():
        local_path.parent.mkdir(parents=True)

    with local_path.open('wb') as f:
        f.write(response.content)

print('finished')
