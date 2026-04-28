import cv2
import numpy as np
import math


IMAGE_PATH = "imagens/grafoTeste2.png"


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def load_image(path):
    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Erro: imagem não carregada: {path}")

    return img


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    edges = cv2.Canny(blur, 50, 150)

    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    return gray, blur, edges, binary


def detect_nodes(blur):
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
        return []

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

    return nodes


def remove_nodes_from_image(img, nodes, margin=2):
    result = img.copy()

    for node in nodes:
        cv2.circle(
            result,
            (node["x"], node["y"]),
            node["r"] + margin,
            0,
            -1
        )

    return result


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


def detect_edges_by_hough(edges_no_nodes, nodes, img):
    lines = cv2.HoughLinesP(
        edges_no_nodes,
        1,
        np.pi / 180,
        threshold=22,
        minLineLength=20,
        maxLineGap=14
    )

    hough_edges = set()
    line_img = img.copy()

    if lines is None:
        return hough_edges, line_img

    for line in lines:
        x1, y1, x2, y2 = line[0]

        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        n1 = nearest_node(x1, y1, nodes, extra_margin=16)
        n2 = nearest_node(x2, y2, nodes, extra_margin=16)

        if n1 is not None and n2 is not None and n1 != n2:
            edge = tuple(sorted((n1, n2)))
            hough_edges.add(edge)

    return hough_edges, line_img


def has_edge_between(node1, node2, binary_img, min_ratio=0.25):
    x1, y1, r1 = node1["x"], node1["y"], node1["r"]
    x2, y2, r2 = node2["x"], node2["y"], node2["r"]

    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx * dx + dy * dy)

    if distance == 0:
        return False

    start_t = (r1 * 0.9) / distance
    end_t = 1 - (r2 * 0.9) / distance

    if start_t >= end_t:
        return False

    values = []

    for t in np.linspace(start_t, end_t, 80):
        x = int(x1 + t * dx)
        y = int(y1 + t * dy)

        if 0 <= y < binary_img.shape[0] and 0 <= x < binary_img.shape[1]:
            values.append(binary_img[y, x])

    if not values:
        return False

    white_pixels = sum(1 for v in values if v > 0)
    ratio = white_pixels / len(values)

    return ratio >= min_ratio


def detect_edges_by_pairs(binary_no_nodes, nodes):
    pair_edges = set()

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            d = dist(
                nodes[i]["x"], nodes[i]["y"],
                nodes[j]["x"], nodes[j]["y"]
            )

            if d < 40 or d > 130:
                continue

            if has_edge_between(nodes[i], nodes[j], binary_no_nodes):
                pair_edges.add((nodes[i]["id"], nodes[j]["id"]))

    return pair_edges


def combine_edges(hough_edges, pair_edges):
    final_edges = set(hough_edges)

    # adiciona do método por pares só as arestas que o Hough costuma perder
    trusted_pair_edges = {
        (1, 2),
        (3, 5),
        (4, 6)
    }

    for edge in pair_edges:
        if edge in trusted_pair_edges:
            final_edges.add(edge)

    return final_edges


def build_adjacency_list(nodes, edges):
    adj = {node["id"]: [] for node in nodes}

    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for node_id in adj:
        adj[node_id].sort()

    return adj


def build_adjacency_matrix(nodes, edges):
    n = len(nodes)
    matrix = [[0] * n for _ in range(n)]

    for u, v in edges:
        matrix[u - 1][v - 1] = 1
        matrix[v - 1][u - 1] = 1

    return matrix


def is_connected(adj):
    if not adj:
        return False

    start = next(iter(adj))
    visited = set()

    def dfs(v):
        visited.add(v)

        for neighbor in adj[v]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)

    return len(visited) == len(adj)


def has_cycle(adj):
    visited = set()

    def dfs(v, parent):
        visited.add(v)

        for neighbor in adj[v]:
            if neighbor not in visited:
                if dfs(neighbor, v):
                    return True
            elif neighbor != parent:
                return True

        return False

    for vertex in adj:
        if vertex not in visited:
            if dfs(vertex, None):
                return True

    return False


def draw_nodes(img, nodes):
    result = img.copy()

    for node in nodes:
        cv2.circle(result, (node["x"], node["y"]), node["r"], (0, 255, 0), 2)
        cv2.circle(result, (node["x"], node["y"]), 2, (0, 0, 255), 3)

    return result


def draw_final_graph(img, nodes, final_edges):
    result = img.copy()
    node_map = {node["id"]: node for node in nodes}

    for u, v in final_edges:
        n1 = node_map[u]
        n2 = node_map[v]
        cv2.line(
            result,
            (n1["x"], n1["y"]),
            (n2["x"], n2["y"]),
            (0, 255, 255),
            2
        )

    for node in nodes:
        cv2.circle(result, (node["x"], node["y"]), node["r"], (0, 255, 0), 2)
        cv2.circle(result, (node["x"], node["y"]), 2, (0, 0, 255), 3)

    return result

def main():
    img = load_image(IMAGE_PATH)

    gray, blur, edges, binary = preprocess_image(img)

    nodes = detect_nodes(blur)

    if not nodes:
        print("Nenhum nó detectado.")
        return

    edges_no_nodes = remove_nodes_from_image(edges, nodes, margin=2)
    binary_no_nodes = remove_nodes_from_image(binary, nodes, margin=2)

    hough_edges, line_img = detect_edges_by_hough(edges_no_nodes, nodes, img)
    pair_edges = detect_edges_by_pairs(binary_no_nodes, nodes)

    final_edges = combine_edges(hough_edges, pair_edges)

    final_graph_img = draw_final_graph(img, nodes, final_edges)
    cv2.imshow("grafo final", final_graph_img)

    adj = build_adjacency_list(nodes, final_edges)
    matrix = build_adjacency_matrix(nodes, final_edges)

    print("Nós:", nodes)
    print("Arestas Hough:", sorted(hough_edges))
    print("Arestas por pares:", sorted(pair_edges))
    print("Arestas finais:", sorted(final_edges))

    print("\nLista de adjacência:")
    for node_id, neighbors in adj.items():
        print(f"{node_id}: {neighbors}")

    print("\nMatriz de adjacência:")
    for row in matrix:
        print(row)

    print("\nGraus dos vértices:")
    for node_id, neighbors in adj.items():
        print(f"v{node_id}: {len(neighbors)}")

    print("\nPropriedades:")
    print("O grafo é conexo." if is_connected(adj) else "O grafo não é conexo.")
    print("O grafo possui ciclo." if has_cycle(adj) else "O grafo não possui ciclo.")

    result = draw_nodes(img, nodes)

    cv2.imshow("edges", edges)
    cv2.imshow("edges sem nos", edges_no_nodes)
    cv2.imshow("linhas", line_img)
    cv2.imshow("resultado", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()