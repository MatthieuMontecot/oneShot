import numpy as np
import exercices as ex


def load_friendly(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        quizzes = np.zeros((9, 9), np.int32)
        for i, line in enumerate(lines):
            quizzes[i] = [int(c) for c in line.split(',')[:9]]
        grid = ex.create_grid(np.transpose(quizzes))
    return grid


def fill_grid_with_s(grid, s):
    assert (len(s) == 81)
    for i in range(9):
        grid[i] = [int(c) for c in s[9 * i: 9 * (i + 1)]]


def from_grid_to_s(grid):
    sum_grid = np.sum(grid, axis=-1)
    s = ''
    for b in range(3):
        for d in range(3):
            for a in range(3):
                for c in range(3):
                    if sum_grid[a, b, c, d] == 1:
                        k = np.where(grid[a, b, c, d] == 1)
                        s += str(k[0][0] + 1)
                    else:
                        s += '0'
    return s


def plot(grid):
    sum_grid = np.sum(grid, axis=-1)
    print('_____________')
    for b in range(3):
        for d in range(3):
            s = ''
            for a in range(3):
                for c in range(3):
                    if sum_grid[a, b, c, d] == 1:
                        k = np.where(grid[a, b, c, d] == 1)
                        s += str(k[0][0] + 1)
                    else:
                        s += 'X'
                if a < 2:
                    s += ','
            print(s)
        if b < 2:
            print('------------')
    print('_____________')


def roll(grid, depth=1):
    """temporary bit"""
    sum_grid_prime = None
    intersection_box_column_counter_array = None  # a (9,9) array about the intersection size between a box number 
    # and a column number 
    intersection_box_line_counter_array = None  # a (9,9) array about the intersection size between a box number and 
    # a line number 
    intersection_box_column_set_array = None  # a (9,9) array about the intersection set between a box number and
    # a column number
    intersection_box_line_set_array = None  # a (9,9) array about the intersection set between a box number and
    # a line number
    loop = True
    total_sum_grid = -5
    is_new_sum_grid = 1
    is_not_static_column = 1
    grid_checkpoint = grid
    while loop:
        loop = False
        while total_sum_grid != np.sum(grid):
            total_sum_grid = np.sum(grid)
            sum_grid = np.sum(grid, axis=-1)
            if 0 in sum_grid:
                break
            if is_new_sum_grid == 1:
                sum_grid_prime = sum_grid
                is_new_sum_grid = 0
                a, b, c, d = np.where(sum_grid == 1)
            else:
                a, b, c, d = np.where(sum_grid * (sum_grid_prime != sum_grid) == 1)
            for f, g, h, i in zip(a, b, c, d):
                k = np.where(grid[f, g, h, i] == 1)[0]
                mask = ex.get_mask(f, g, h, i, k)
                grid = grid * mask
            detect_on_columns = np.sum(grid, axis=(0, 2))
            boxes_lines, lines, ks = np.where(detect_on_columns == 1)
            for yBox, k, line in zip(boxes_lines.flatten(), ks.flatten(), lines.flatten()):
                boxes_abscissa, abscissa, = np.where(grid[:, yBox, :, line, k] == 1)
                if len(boxes_abscissa) != 0:
                    mask = ex.get_mask(boxes_abscissa[0], yBox, abscissa[0], line, k)
                    grid = grid * mask

            detect_on_lines = np.sum(grid, axis=(1, 3))
            boxes_columns, columns, ks = np.where(detect_on_lines == 1)
            for xBox, k, column in zip(boxes_columns.flatten(), ks.flatten(), columns.flatten()):
                boxes_column, ys, = np.where(grid[xBox, :, column, :, k] == 1)
                if len(boxes_column) != 0:
                    mask = ex.get_mask(xBox, boxes_column[0], column, ys[0], k)
                    grid = grid * mask
            if is_not_static_column:
                intersection_box_line_set_array = np.empty((9, 9), dtype=set)
                intersection_box_column_set_array = np.empty((9, 9), dtype=set)
                intersection_box_line_counter_array = np.empty((9, 9), dtype='int32')
                intersection_box_column_counter_array = np.empty((9, 9), dtype='int32')
                is_not_static_column = 0
            for Bx in range(3):
                for By in range(3):
                    for Lx in range(3):
                        flat_column_number = 3 * Bx + Lx
                        flat_box_number = Bx + 3 * By
                        temporary_sub_column_array = grid[Bx, By, Lx]
                        temporary_filter = np.sum(temporary_sub_column_array, axis=-1).reshape((3, 1))
                        temporary_sub_column_array = temporary_sub_column_array * (temporary_filter != 1)
                        j, k = np.where(temporary_sub_column_array == 1)
                        j = set(j)
                        possibilities = set(k)
                        intersection_box_column_counter_array[flat_box_number, flat_column_number] = len(j)
                        intersection_box_column_set_array[flat_box_number, flat_column_number] = possibilities
                    for Ly in range(3):
                        flat_line_number = 3 * By + Ly
                        flat_box_number = Bx + 3 * By
                        temporary_sub_line_array = grid[Bx, By, :, Ly]
                        temporary_filter = np.sum(temporary_sub_line_array, axis=-1).reshape((3, 1))
                        temporary_sub_line_array = temporary_sub_line_array * (temporary_filter != 1)
                        j, k = np.where(temporary_sub_line_array == 1)
                        j = set(j)
                        possibilities = set(k)
                        intersection_box_line_counter_array[flat_box_number, flat_line_number] = len(j)
                        intersection_box_line_set_array[flat_box_number, flat_line_number] = possibilities

            sum_grid = np.sum(grid, axis=-1)
            if 0 in sum_grid:
                break
            for Bx in range(3):
                for By in range(3):
                    for Lx in range(3):
                        flat_column_number = 3 * Bx + Lx
                        flat_box_number = Bx + 3 * By
                        if intersection_box_column_counter_array[flat_box_number,
                                                                 flat_column_number] == \
                                len(intersection_box_column_set_array[flat_box_number, flat_column_number]):
                            y_coordinates, = np.where(sum_grid[Bx, By, Lx] != 1)
                            if len(y_coordinates) == 0 or \
                                    len(intersection_box_column_set_array[flat_box_number, flat_column_number]) == 0:
                                continue
                            grid[np.ix_([Bx], [By], range(3), range(3),
                                        list(intersection_box_column_set_array[flat_box_number,
                                                                               flat_column_number]))] = np.zeros(
                                (1, 1, 3, 3, len(list(intersection_box_column_set_array[flat_box_number,
                                                                                        flat_column_number]))))
                            grid[np.ix_([Bx], range(3), [Lx], range(3),
                                        list(intersection_box_column_set_array[flat_box_number,
                                                                               flat_column_number]))] = np.zeros(
                                (1, 3, 1, 3, len(list(intersection_box_column_set_array[flat_box_number,
                                                                                        flat_column_number]))))
                            grid[np.ix_([Bx], [By], [Lx], y_coordinates,
                                        list(intersection_box_column_set_array[flat_box_number,
                                                                               flat_column_number]))] = np.ones(
                                [len(y_coordinates),
                                 len(list(intersection_box_column_set_array[flat_box_number,
                                                                            flat_column_number]))]).reshape(
                                (1, 1, 1, len(y_coordinates),
                                 len(list(intersection_box_column_set_array[flat_box_number, flat_column_number]))))
                    for Ly in range(3):
                        flat_line_number = 3 * By + Ly
                        flat_box_number = Bx + 3 * By
                        if intersection_box_line_counter_array[flat_box_number,
                                                               flat_line_number] == \
                                len(intersection_box_line_set_array[flat_box_number, flat_line_number]):
                            x_coordinates, = np.where(sum_grid[Bx, By, :, Ly] != 1)
                            if len(x_coordinates) == 0 or len(intersection_box_line_set_array[flat_box_number,
                                                                                              flat_line_number]) == 0:
                                continue
                            grid[np.ix_([Bx], [By], range(3), range(3),
                                        list(intersection_box_line_set_array[flat_box_number,
                                                                             flat_line_number]))] = np.zeros(
                                (1, 1, 3, 3, len(list(intersection_box_line_set_array[flat_box_number,
                                                                                      flat_line_number]))))
                            grid[np.ix_(range(3), [By], range(3), [Ly],
                                        list(intersection_box_line_set_array[flat_box_number,
                                                                             flat_line_number]))] = np.zeros(
                                (3, 1, 3, 1, len(list(intersection_box_line_set_array[flat_box_number,
                                                                                      flat_line_number]))))
                            grid[np.ix_([Bx], [By], x_coordinates, [Ly],
                                        list(intersection_box_line_set_array[flat_box_number,
                                                                             flat_line_number]))] = np.ones(
                                [len(x_coordinates),
                                 len(list(intersection_box_line_set_array[flat_box_number,
                                                                          flat_line_number]))]).reshape(
                                (1, 1, len(x_coordinates), 1,
                                    len(list(intersection_box_line_set_array[flat_box_number, flat_line_number]))))
        sum_grid = np.sum(grid, axis=-1)
        detect_on_columns = np.sum(grid, axis=(0, 2))
        detect_on_lines = np.sum(grid, axis=(1, 3))
        for i in range(3):
            for j in range(3):
                sum_over_boxes_array = np.sum(grid[i, j], axis=(0, 1))
                if 0 in sum_over_boxes_array:
                    return -1, grid_checkpoint
        if 0 in sum_grid or 0 in detect_on_columns or 0 in detect_on_lines:
            return -1, grid_checkpoint
        if depth == 0:
            return 0, grid_checkpoint
        dirty_grid = grid
        sum_grid = np.sum(grid, axis=-1)
        hypothesis = np.where(sum_grid != 1)
        box_column_hypothesis, box_line_hypothesis, sub_column_hypothesis, sub_line_hypothesis = hypothesis
        box_column_hypothesis = list(box_column_hypothesis)
        box_line_hypothesis = list(box_line_hypothesis)
        sub_column_hypothesis = list(sub_column_hypothesis)
        sub_line_hypothesis = list(sub_line_hypothesis)
        values = sum_grid[hypothesis[0], hypothesis[1], hypothesis[2], hypothesis[3]].tolist()
        while len(values) != 0:
            x_min = np.argmin(values)
            a = box_column_hypothesis.pop(x_min)
            b = box_line_hypothesis.pop(x_min)
            c = sub_column_hypothesis.pop(x_min)
            d = sub_line_hypothesis.pop(x_min)
            values.pop(x_min)
            possible_key_list = np.where(grid[a, b, c, d] == 1)[0]
            for key in possible_key_list:
                mask = ex.get_mask(a, b, c, d, key)
                dirty_grid = dirty_grid * mask
                worked, next_step_grid = roll(dirty_grid, depth - 1)
                if worked == 1:
                    return worked, next_step_grid
                if worked == -1:
                    grid[a, b, c, d, key] = 0
                    values = []
                    loop = True
                    break
    return 0, grid
