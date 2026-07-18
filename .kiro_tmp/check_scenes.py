import os

scenes = sorted(os.listdir('data/octscenes'))
total = 0
ready = 0
for s in scenes:
    if not os.path.isdir(f'data/octscenes/{s}'):
        continue
    total += 1
    splat = f'data/octscenes/{s}/splat.ply'
    if os.path.exists(splat):
        size = os.path.getsize(splat) / (1024*1024)
        print(f'  {s}: splat.ply ({size:.1f} MB)')
        ready += 1
    else:
        print(f'  {s}: NO splat.ply')

print(f'\nTotal: {ready}/{total} scenes have splat.ply')
