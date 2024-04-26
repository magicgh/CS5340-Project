import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20

data = {
    'size': ['1.4b', '2.8b'],
    #EU SST: [mean, var]
    # prompt 1
    'mamba': [[0.4503, 0.2401], [0.4277, 0.2553]],
    'mamba-lima': [[0.0042, 0.0188], [0.1327, 0.1061]],
    'llama': [0.0274, 0.1209],
    'llama-lima': [0.1050, 0.0704],
    'gemma': [0.0795, 0.0709],
    'gemma-lima': [0.0386, 0.0330]

    # prompt 2
    # 'mamba': [[0.3974, 0.2377], [0.3798, 0.2372]],
    # 'mamba-lima': [[0.0161, 0.0313], [0.1087, 0.0948]],
    # 'llama': [0.1223, 0.2400],
    # 'llama-lima': [0.0121, 0.0104],
    # 'gemma': [0.1134, 0.0958],
    # 'gemma-lima': [0.0299, 0.0248]

    # EU financial: [mean, var]
    # prompt 1
    # 'mamba': [[0.5438, 0.2549], [0.5104, 0.2769]],
    # 'mamba-lima': [[0.0299, 0.0768], [0.1678, 0.1204]],
    # 'llama': [0.5739, 0.2820],
    # 'llama-lima': [0.0461, 0.0666],
    # 'gemma': [0.2976, 0.1866],
    # 'gemma-lima': [0.4286, 0.1996]

    # prompt 2
    # 'mamba': [[0.6532, 0.2664], [0.5462, 0.2765]],
    # 'mamba-lima': [[0.0079, 0.0403], [0.0898, 0.0917]],
    # 'llama': [0.5468, 0.2877],
    # 'llama-lima': [0.0779, 0.0653],
    # 'gemma': [0.3409, 0.2061],
    # 'gemma-lima': [0.2474, 0.1759]

}

type_colors = {'p1': 'blue', 'p2': 'orange', 'p3': ''}

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(['before-lima', 'after-lima'], [data['mamba'][0][0], data['mamba-lima'][0][0]], yerr=[data['mamba'][0][1], data['mamba-lima'][0][1]], capsize=5, label='mamba-1.4b', marker='o')
ax.errorbar(['before-lima', 'after-lima'], [data['mamba'][1][0], data['mamba-lima'][1][0]], yerr=[data['mamba'][1][1], data['mamba-lima'][1][1]], capsize=5, label='mamba-2.8b', marker='o')
ax.errorbar(['before-lima', 'after-lima'], [data['llama'][0], data['llama-lima'][0]], yerr=[data['llama'][1], data['llama-lima'][1]], capsize=5, label='llama2-7b', marker='o')
ax.errorbar(['before-lima', 'after-lima'], [data['gemma'][0], data['gemma-lima'][0]], yerr=[data['gemma'][1], data['gemma-lima'][1]], capsize=5, label='gemma-2b', marker='o')

# ax.set_xlabel('Models')
ax.set_ylabel('EU Mean')
ax.set_title('EU Results for Training Comparison (sst Dataset)')


plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('EU Results for Training Comparison on the sst prompt1.png')
plt.show()
