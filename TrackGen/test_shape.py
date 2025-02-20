import matplotlib.pyplot as plt
from shapely.affinity import rotate
from shapely.geometry import box, Polygon


def gen_shape():
    rect = box(0, 0, 2, 1)

    box = box(0, 0, 1, 1)
    rotated_rect = rotate(box, 10, origin='center')

    intersection = rect.intersection(rotated_rect)
    print(box)

    x, y = rect.exterior.xy
    xr, yr = rotated_rect.exterior.xy
    xi, yi = intersection.exterior.xy

    print(xi, yi)
    assert 1 == 2
    plt.figure()
    plt.plot(x, y, label='Original Rectangle', color='blue')
    plt.plot(xr, yr, label='Rotated Rectangle', color='red')
    plt.fill(xi, yi, label='Intersection', color='green', alpha=0.5)
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rectangle Rotation and Intersection')
    plt.axis('equal')
    plt.show()


def create_grid(x_min, x_max, y_min, y_max, cell_size):
    grid = []
    x = x_min
    while x < x_max:
        y = y_min
        while y < y_max:
            cell = Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
            grid.append(cell)
            y += cell_size
        x += cell_size
    return grid


x_min, x_max, y_min, y_max = 0, 1000, 0, 1000
cell_size = 1

rect = box(3, 3, 7, 6)

rotated_rect = rotate(rect, 45, origin='center')

print("======1=====")

intersection = rect.intersection(rotated_rect)
print("======2=====")

grid = create_grid(x_min, x_max, y_min, y_max, cell_size)
print("======3=====")

fig, ax = plt.subplots()

x, y = rect.exterior.xy
ax.plot(x, y, color='blue', label='Original Rectangle')

xr, yr = rotated_rect.exterior.xy
ax.plot(xr, yr, color='red', label='Rotated Rectangle')

for cell in grid:
    x, y = cell.exterior.xy
    intersection_part = cell.intersection(intersection)
    if not intersection_part.is_empty:
        if intersection_part.geom_type == 'Polygon':
            xi, yi = intersection_part.exterior.xy
            ax.fill(xi, yi, color='yellow', alpha=0.5)
        elif intersection_part.geom_type == 'MultiPolygon':
            for part in intersection_part:
                xi, yi = part.exterior.xy
                ax.fill(xi, yi, color='yellow', alpha=0.5)
    ax.plot(x, y, color='black', linewidth=0.5)

ax.set_aspect('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grid with Intersection Highlighted')
plt.legend()
plt.show()
