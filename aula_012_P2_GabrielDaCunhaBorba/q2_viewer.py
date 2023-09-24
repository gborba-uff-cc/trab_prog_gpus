import json
import matplotlib
import matplotlib.pyplot as plt
import sys

# def readJSON(filename):
#     data = json.load(open(filename))
#     return data

# res = readJSON(sys.argv[1])
# # plt.plot(res_prof['resultado'], label='prof')
# plt.plot(res['resultado'], label='Json')
# plt.legend(loc='best')
# plt.show()

for nomeJson in sys.argv[1:]:
    data = []
    with open(nomeJson, 'r') as arq:
        data = json.load(arq)
    plt.plot(data['resultado'], label=nomeJson)

    plt.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')
    plt.savefig(nomeJson.replace('.json', '.png'))
    plt.clf()

# plt.show()
