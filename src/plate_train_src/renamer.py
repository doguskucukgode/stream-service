import os

p = "/home/taylan/local/plate_data/train"

data = os.path.abspath(p)
for i, f in enumerate(os.listdir(p)):
    src = os.path.join(data, f)
    parts = f.split('_')
    plate = parts[0]
    uuid_stuff = parts[1]

    flag = True
    new_plate = ''
    for c in plate:
        if flag:
            if not c.isalpha():
                new_plate += c
            else:
                flag = 1
                new_plate += ' '
                new_plate += c
        else:
            if c.isalpha():
                new_plate += c
            else:
                flag = False
                new_plate += ' '
                new_plate += c

    new_plate += '_'
    new_plate += uuid_stuff

    dst = os.path.join(data, new_plate)
    print(src)
    print(dst)
    os.rename(src, dst)
