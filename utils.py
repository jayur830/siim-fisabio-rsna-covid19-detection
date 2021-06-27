def convert_to_yolo(grid_width: int, grid_height: int, x: float, y: float, w: float, h: float):
    grid_x, grid_y = int(x * grid_width), int(y * grid_height)
    x, y = x * grid_width - grid_x, y * grid_height - grid_y
    return grid_x, grid_y, x, y, w, h


def convert_to_real(
        target_width: int,
        target_height: int,
        grid_width: int,
        grid_height: int,
        anchor_width: float,
        anchor_height: float,
        c_x: int,
        c_y: int,
        t_x: float,
        t_y: float,
        t_w: float,
        t_h: float):
    b_x = (t_x + c_x) * target_width / grid_width
    b_y = (t_y + c_y) * target_height / grid_height
    b_w = t_w * target_width * anchor_width / grid_width
    b_h = t_h * target_height * anchor_height / grid_height
    return int(b_x - b_w * .5), int(b_y - b_h * .5), int(b_x + b_w * .5), int(b_y + b_h * .5)