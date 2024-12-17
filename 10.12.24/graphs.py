import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

k = 2
n = 10
e = n * n * np.random.randint(20, 40) // 100
G: nx.Graph = nx.random_graphs.gnm_random_graph(n, e)

# Заполнение весами
for u, v in G.edges():
    G[u][v]['weight'] = np.random.randint(10, 100)

# Генерируем позиции для узлов
pos = nx.spring_layout(G)

# Строим минимальное остовное дерево
mst = nx.minimum_spanning_tree(G)

# Рисуем граф
nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=15)

# Получаем веса рёбер
edge_labels = nx.get_edge_attributes(G, "weight")

# Добавляем текстовые метки весов на рёбра
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", font_size=12)

# Остовное дерево
nx.draw(mst, pos, with_labels=True, edge_color="red", width=3, node_size=500, style='dashed')

longest_edges = sorted(list(mst.edges(data=True)),  key= lambda e: e[2]['weight'], reverse=True)[:2]

mst.remove_edges_from([(u, v) for (u, v, _) in longest_edges])

# Остовное дерево
nx.draw(mst, pos, with_labels=True, edge_color="blue", width=3, node_size=500, style='dashed')

# Показываем граф
plt.show()