par = [
    (1, 1, 1),
    (2, 1, 1),
    (3, 1, 1),
    (2, 2, 1),
    (5, 1, 1),
    (3, 2, 1),
    (7, 1, 1),
    (2, 2, 2),
    (3, 3, 1),
    (5, 2, 1),
    (11, 1, 1),
    (3, 2, 2),
    (13, 1, 1),
    (7, 2, 1),
    (5, 3, 1),
    (4, 2, 2),
    (17, 1, 1),
    (3, 3, 2),
    (19, 1, 1),
    (5, 2, 2),
    (7, 3, 1),
    (11, 2, 1),
    (23, 1, 1),
    (4, 3, 2),
    (5, 5, 1),
    (13, 2, 1),
    (3, 3, 3),
    (7, 2, 2),
    (29, 1, 1),
    (5, 3, 2)
]

with open('run-template.sh', 'r') as rd:
    inp = rd.readlines()

t = ''
for s in inp:
    t += s

for p in par:
    tasks = p[0] * p[1] * p[2]
    file = f'out-{tasks}.txt'
    with open(f'run-{tasks}.sh', 'w') as wr:
        wr.write(t
            .replace('$${tasks}', f'{tasks}')
            .replace('$${node}', f'{tasks if tasks < 16 else 16}')
            .replace('$${x}', f'{p[0]}')
            .replace('$${y}', f'{p[1]}')
            .replace('$${z}', f'{p[2]}')
            .replace('$${file}', file)
        )