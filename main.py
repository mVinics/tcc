import cv2
import numpy as np
import math

img = cv2.imread("imagens/grafoTeste2.png")

if img is None:
    print("Erro: imagem não carregada")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 2)
edges = cv2.Canny(blur, 50, 150)

circles = cv2.HoughCircles(
    blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=25,
    param1=50,
    param2=18,
    minRadius=8,
    maxRadius=16
)

if circles is None:
    print("Nenhum círculo detectado")
    exit()

circles = np.uint16(np.around(circles))

nodes = []
for idx, (x, y, r) in enumerate(circles[0], start=1):
    nodes.append({
        "id": idx,
        "x": int(x),
        "y": int(y),
        "r": int(r)
    })

nodes.sort(key=lambda n: (n["y"], n["x"]))
for i, node in enumerate(nodes, start=1):
    node["id"] = i

print("Nós:", nodes)

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def nearest_node(px, py, nodes, extra_margin=10):
    best_node = None
    best_d = float("inf")

    for node in nodes:
        d = dist(px, py, node["x"], node["y"])
        limit = node["r"] + extra_margin
        if d <= limit and d < best_d:
            best_d = d
            best_node = node["id"]

    return best_node

# remove os vértices da imagem de bordas
edges_no_nodes = edges.copy()
for node in nodes:
    cv2.circle(edges_no_nodes, (node["x"], node["y"]), node["r"] + 2, 0, -1)



#verificação de linha preta
def has_edge(node1, node2, edges_img):
    x1, y1 = node1["x"], node1["y"]
    x2, y2 = node2["x"], node2["y"]

    # amostrar pontos ao longo da linha
    for t in np.linspace(0, 1, 20):
        x = int(x1 + t*(x2 - x1))
        y = int(y1 + t*(y2 - y1))

        if edges_img[y, x] > 0:
            return True

    return False

edges_list = set()

for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        if has_edge(nodes[i], nodes[j], edges):
            edges_list.add((nodes[i]["id"], nodes[j]["id"]))


#verificação por par de nós

def sample_line_pixels(img, x1, y1, x2, y2, steps=100):
    points = []
    for t in np.linspace(0, 1, steps):
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            points.append(img[y, x])
    return points


def has_edge_between(node1, node2, edge_img, ignore_radius_factor=0.9, min_ratio=0.35):
    x1, y1, r1 = node1["x"], node1["y"], node1["r"]
    x2, y2, r2 = node2["x"], node2["y"], node2["r"]

    dx = x2 - x1
    dy = y2 - y1
    dist_nodes = math.sqrt(dx * dx + dy * dy)
    if dist_nodes == 0:
        return False

    start_t = (r1 * ignore_radius_factor) / dist_nodes
    end_t = 1 - (r2 * ignore_radius_factor) / dist_nodes

    if start_t >= end_t:
        return False

    values = []
    for t in np.linspace(start_t, end_t, 80):
        x = int(x1 + t * dx)
        y = int(y1 + t * dy)
        if 0 <= y < edge_img.shape[0] and 0 <= x < edge_img.shape[1]:
            values.append(edge_img[y, x])

    if not values:
        return False

    white_pixels = sum(1 for v in values if v > 0)
    ratio = white_pixels / len(values)

    return ratio >= min_ratio



#binary 

_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

binary_no_nodes = binary.copy()
for node in nodes:
    cv2.circle(binary_no_nodes, (node["x"], node["y"]), node["r"] + 2, 0, -1)

pair_edges = set()

for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):
        if has_edge_between(nodes[i], nodes[j], binary_no_nodes, min_ratio=0.25):
            pair_edges.add((nodes[i]["id"], nodes[j]["id"]))

print("Arestas por pares:", sorted(pair_edges))


final_edges = set()



# ----------------------------
# 1) Arestas por pares
# ----------------------------

pair_edges = set()

for i in range(len(nodes)):
    for j in range(i + 1, len(nodes)):

        # calcula distância entre nós
        d = dist(nodes[i]["x"], nodes[i]["y"],
                 nodes[j]["x"], nodes[j]["y"])

        # filtro de distância
        if d < 40 or d > 130:
            continue

        # verificação de aresta
        if has_edge_between(nodes[i], nodes[j], binary_no_nodes, min_ratio=0.25):
            pair_edges.add((nodes[i]["id"], nodes[j]["id"]))


# ----------------------------
# 2) Arestas por Hough
# ----------------------------    

# detecta linhas sem os nós
lines = cv2.HoughLinesP(
    edges_no_nodes,
    1,
    np.pi / 180,
    threshold=22,
    minLineLength=20,
    maxLineGap=14
)

line_img = img.copy()
edges_list = set()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]

        n1 = nearest_node(x1, y1, nodes, extra_margin=16)
        n2 = nearest_node(x2, y2, nodes, extra_margin=16)

        if n1 is not None and n2 is not None and n1 != n2:
            edge = tuple(sorted((n1, n2)))
            edges_list.add(edge)
            print(f"Linha ({x1},{y1}) -> ({x2},{y2}) => {edge}")

print("Arestas:", sorted(edges_list))


# ----------------------------
# 3) Combinação final
# ----------------------------

# tudo que o Hough acha
final_edges.update(edges_list)

# só adiciona do método por pares se passar filtros
for edge in pair_edges:
    if edge in edges_list:
        continue  # já confirmado
    
    # aqui você pode aplicar filtros extras
    final_edges.add(edge)

print("Arestas finais:", sorted(final_edges))



#lista de adjacencia
adj = {node["id"]: [] for node in nodes}

for u, v in final_edges:
    adj[u].append(v)
    adj[v].append(u)

for k in adj:
    adj[k].sort()

print("Lista de adjacência:")
for k, v in adj.items():
    print(f"{k}: {v}")


#matriz de adjacencia
n = len(nodes)
matrix = [[0] * n for _ in range(n)]

for u, v in final_edges:
    matrix[u - 1][v - 1] = 1
    matrix[v - 1][u - 1] = 1

print("Matriz de adjacência:")
for row in matrix:
    print(row)


print("Graus dos vértices:")
for node_id in adj:
    print(f"v{node_id}: {len(adj[node_id])}")

# desenha nós detectados
result = img.copy()
for node in nodes:
    cv2.circle(result, (node["x"], node["y"]), node["r"], (0, 255, 0), 2)
    cv2.circle(result, (node["x"], node["y"]), 2, (0, 0, 255), 3)

cv2.imshow("edges", edges)
cv2.imshow("edges sem nos", edges_no_nodes)
cv2.imshow("linhas", line_img)
cv2.imshow("resultado", result)
cv2.waitKey(0)
cv2.destroyAllWindows()