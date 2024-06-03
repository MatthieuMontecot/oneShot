import numpy as np


def get_mask(a, b, c, d, k, init=False):
    mask = np.ones((3, 3, 3, 3, 9))
    if not init:
        mask[a, :, c, :, k] = np.zeros((3, 3))
        mask[:, b, :, d, k] = np.zeros((3, 3))
        mask[a, b, :, :, k] = np.zeros((3, 3))
    mask[a, b, c, d, :] = np.zeros(9)
    mask[a, b, c, d, k] = 1
    return mask


def from_grid_to_array(grid):
    readable_grid = np.empty((9, 9), dtype='object')
    sum_grid = np.sum(grid, axis=-1)
    for i in range(9):
        for j in range(9):
            if sum_grid[i // 3, j // 3, i % 3, j % 3] == 1:
                readable_grid[j, i] = str(np.where(grid[i // 3, j // 3, i % 3, j % 3] == 1)[0][0] + 1)
            elif sum_grid[i // 3, j // 3, i % 3, j % 3] == 0:
                readable_grid[j, i] = 'O'
            else:
                readable_grid[j, i] = 'X'
    if 0 in np.sum(grid, axis=(0, 2)):
        print('pb line')
    if 0 in np.sum(grid, axis=(1, 3)):
        print('pb column')
    if 0 in np.sum(grid, axis=(2, 3)):
        print('pb box')
    return np.transpose(readable_grid)


def check(readable_grid):
    if 'X' in readable_grid or 'O' in readable_grid:
        return False
    ref = {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
    for i in range(9):
        if set(readable_grid[i, :].tolist()) != ref:
            print('error in line', i)
            return False
        if set(readable_grid[:, i].tolist()) != ref:
            print('error in column', i)
            return False
    for i in range(3):
        for j in range(3):
            if set(readable_grid[3 * i:3 * i + 3, 3 * j:3 * j + 3].flatten().tolist()) != ref:
                print('error in box', i, j)
                return False
    return True


def create_grid(readable_grid):
    ref = set(range(9))
    grid = np.ones((3, 3, 3, 3, 9))
    for i in range(9):
        for j in range(9):
            value = readable_grid[i, j] - 1
            if value in ref:
                grid[i // 3, j // 3, i % 3, j % 3, list(ref - {value})] = np.zeros(8)
    return grid


def exo2():
    readable_grid = np.array(((3, 4, 0, 2, 0, 9, 5, 6, 1),
                              (9, 0, 6, 5, 1, 4, 0, 3, 2),
                              (1, 2, 0, 8, 3, 0, 7, 4, 9),
                              (0, 5, 3, 6, 2, 1, 9, 0, 4),
                              (0, 8, 2, 0, 9, 7, 0, 5, 3),
                              (4, 9, 0, 3, 8, 5, 2, 7, 0),
                              (2, 0, 4, 1, 0, 0, 3, 9, 0),
                              (8, 1, 7, 0, 6, 3, 4, 2, 5),
                              (5, 0, 9, 7, 4, 2, 0, 1, 8)))
    return create_grid(readable_grid)


def exo3():
    readable_grid = np.array(((0, 4, 0, 3, 0, 0, 0, 0, 0),
                              (0, 8, 0, 5, 0, 0, 2, 0, 0),
                              (0, 0, 1, 2, 0, 0, 0, 8, 0),
                              (0, 0, 2, 0, 0, 0, 0, 6, 9),
                              (0, 0, 0, 0, 0, 0, 0, 1, 7),
                              (0, 0, 5, 0, 4, 0, 0, 0, 0),
                              (0, 0, 0, 0, 5, 0, 9, 0, 3),
                              (0, 1, 0, 0, 3, 9, 0, 7, 0),
                              (6, 0, 0, 0, 0, 0, 0, 0, 0)))
    return create_grid(readable_grid)


def exo4():
    readable_grid = np.array(((0, 0, 0, 8, 0, 1, 0, 0, 0),
                              (0, 0, 0, 0, 0, 0, 0, 4, 3),
                              (5, 0, 0, 0, 0, 0, 0, 0, 0),
                              (0, 0, 0, 0, 7, 0, 8, 0, 0),
                              (0, 0, 0, 0, 0, 0, 1, 0, 0),
                              (0, 2, 0, 0, 3, 0, 0, 0, 0),
                              (6, 0, 0, 0, 0, 0, 0, 7, 5),
                              (0, 0, 3, 4, 0, 0, 0, 0, 0),
                              (0, 0, 0, 2, 0, 0, 6, 0, 0)))
    return create_grid(readable_grid)


def exo5():
    readable_grid = np.array(((0, 0, 0, 0, 0, 0, 0, 0, 1),
                              (0, 0, 0, 0, 0, 0, 0, 2, 3),
                              (0, 0, 4, 0, 0, 5, 0, 0, 0),
                              (0, 0, 0, 1, 0, 0, 0, 0, 0),
                              (0, 0, 0, 0, 3, 0, 6, 0, 0),
                              (0, 0, 7, 0, 0, 0, 5, 8, 0),
                              (0, 0, 0, 0, 6, 7, 0, 0, 0),
                              (0, 1, 0, 0, 0, 4, 0, 0, 0),
                              (5, 2, 0, 0, 0, 0, 0, 0, 0)))
    return create_grid(readable_grid)


def exo6():
    readable_grid = np.array(((8, 0, 0, 0, 0, 0, 0, 0, 0),
                              (0, 0, 3, 6, 0, 0, 0, 0, 0),
                              (0, 7, 0, 0, 9, 0, 2, 0, 0),
                              (0, 5, 0, 0, 0, 7, 0, 0, 0),
                              (0, 0, 0, 0, 4, 5, 7, 0, 0),
                              (0, 0, 0, 1, 0, 0, 0, 3, 0),
                              (0, 0, 1, 0, 0, 0, 0, 6, 8),
                              (0, 0, 8, 5, 0, 0, 0, 1, 0),
                              (0, 9, 0, 0, 0, 0, 4, 0, 0)))
    return create_grid(readable_grid)


def exo7():
    readable_grid = np.array(((0, 2, 8, 0, 1, 9, 0, 0, 4),
                              (0, 0, 1, 0, 0, 0, 0, 0, 0),
                              (0, 6, 0, 0, 0, 0, 5, 0, 0),
                              (0, 0, 4, 0, 5, 0, 0, 7, 0),
                              (0, 3, 0, 4, 0, 1, 0, 8, 0),
                              (0, 8, 0, 0, 3, 0, 2, 0, 0),
                              (0, 0, 9, 0, 0, 0, 0, 6, 0),
                              (0, 0, 0, 0, 0, 0, 9, 0, 0),
                              (4, 0, 0, 2, 9, 0, 3, 5, 0)))
    return create_grid(readable_grid)


def exo1():
    grid = np.ones((3, 3, 3, 3, 9)).astype('int32')
    mask = getMask(0, 0, 0, 0, 7, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 1, 0, 6, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 0, 1, 5, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 1, 1, 0, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 2, 1, 3, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 1, 2, 1, init=True)
    grid = grid * mask
    mask = getMask(0, 0, 2, 2, 2, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 0, 0, 0, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 1, 0, 1, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 2, 1, 4, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 0, 2, 7, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 1, 2, 3, init=True)
    grid = grid * mask
    mask = getMask(1, 0, 2, 2, 5, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 0, 0, 5, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 1, 0, 3, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 2, 0, 4, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 0, 1, 1, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 2, 1, 7, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 1, 2, 8, init=True)
    grid = grid * mask
    mask = getMask(2, 0, 2, 2, 0, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 0, 0, 8, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 1, 0, 3, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 0, 1, 0, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 2, 1, 4, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 0, 2, 6, init=True)
    grid = grid * mask
    mask = getMask(0, 1, 1, 2, 7, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 1, 0, 7, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 2, 0, 6, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 0, 1, 1, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 2, 1, 3, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 0, 2, 2, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 1, 2, 0, init=True)
    grid = grid * mask
    mask = getMask(1, 1, 2, 2, 8, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 0, 0, 2, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 1, 0, 0, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 0, 1, 8, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 2, 1, 6, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 1, 2, 4, init=True)
    grid = grid * mask
    mask = getMask(2, 1, 2, 2, 1, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 2, 0, 6, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 0, 1, 2, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 1, 1, 8, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 2, 1, 7, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 0, 2, 3, init=True)
    grid = grid * mask
    mask = getMask(0, 2, 2, 2, 0, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 0, 0, 3, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 1, 0, 2, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 2, 0, 0, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 0, 1, 5, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 1, 1, 4, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 1, 2, 6, init=True)
    grid = grid * mask
    mask = getMask(1, 2, 2, 2, 7, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 0, 0, 7, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 1, 0, 5, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 0, 1, 0, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 2, 1, 3, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 0, 2, 4, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 1, 2, 1, init=True)
    grid = grid * mask
    mask = getMask(2, 2, 2, 2, 2, init=True)
    grid = grid * mask
    return grid
