import os
import json
import os.path
import matplotlib.pyplot as plt
from alisuretool.Tools import Tools


# https://github.com/yangyangyang127/APE/blob/main/APE_results.ipynb
# https://blog.csdn.net/HLBoy_happy/article/details/131667829


class ReadResult(object):

    def __init__(self, txt_path="./result/log/1_main_experiment_2_few_shot.txt"):
        with open(txt_path, "r") as f:
            result = f.readlines()
        self.result = [one for one in result if "name" in one]
        pass

    def get_result(self):
        results = []

        # 解析结果
        for one in self.result:
            now_result = " ".join(one.split(" ")[2:]).replace("'", "\"")
            json_result = json.loads(now_result)
            if "imagenet" in one:
                results.append({})
                results[-1][json_result["name"]] = [max([two[key] for key in two]) for two in json_result["acc"]]
                results[-1]["detail"] = json_result["detail"]
                pass
            else:
                results[-1][json_result["name"]] = [max([two[key] for key in two]) for two in json_result["acc"]]
                pass

        # 平均结果
        result_avg = []
        for one in results:
            num = [one[key][0] for key in one if "detail" not in key]
            result_avg.append(sum(num)/len(num))
            pass

        # 打印结果
        for idx, one in enumerate(results):
            detail = one["detail"]
            del one["detail"]
            print(detail)
            for key in one:
                now_r = " ".join([f"{two:.2f}" for two in one[key]])
                print(f"{key}: {now_r}")
            one["detail"] = detail
            print(result_avg[idx])
            pass

        # 得到最终结果
        print()
        all_key = [key for key in results[0] if "detail" not in key]
        now_result = [[one[key][0] for one in results] for key in all_key]
        for i in range(len(now_result[0]) // 5):
            this_result = [one[i * 5: i * 5 + 5] for one in now_result]
            for idx, one in enumerate(this_result):
                print(f"{all_key[idx]} {one}")
        return results, result_avg

    pass


class DataAndMethod(object):

    caltech101 = {"other": {'CoOp': [87.8, 87.87, 89.57, 90.1, 92.07],
                            # 'LP-CLIP': [70.62, 78.72, 84.34, 87.78, 90.63],
                            'Proto-CLIP': [87.99, 89.05, 89.57, 90.22, 91.08],
                            'Proto-CLIP-F': [88.07, 89.09, 89.62, 92.54, 93.43],
                            'CLIP-Adapter': [88.63, 89.42, 89.98, 91.95, 92.39],
                            'SuS-X': [88.39, 88.6, 89.29, 89.81, 90.7],
                            'Tip-Adapter': [87.26, 88.32, 89.21, 89.57, 90.18],
                            'Tip-Adapter-F': [89.21, 89.94, 91.12, 92.01, 92.86],
                            'APE': [90.47, 90.99, 91.81, 91.93, 92.49]},
                  "zs": 85.84,
                  "our": [89.73630831643003, 90.22312373225152, 90.8316430020284, 92.61663286004057, 93.63083164300204]}

    dtd = {"other": {'CoOp': [44.1, 46.0, 53.53, 59.63, 63.7],
                     # 'LP-CLIP': [29.59, 39.48, 50.06, 56.56, 63.97],
                     'Proto-CLIP': [46.04, 51.06, 55.91, 59.34, 61.64],
                     'Proto-CLIP-F': [35.64, 49.88, 57.21, 62.35, 68.56],
                     'CLIP-Adapter': [45.87, 52.13, 57.07, 61.33, 65.03],
                     'SuS-X': [46.98, 50.11, 55.2, 59.33, 63.53],
                     'Tip-Adapter': [46.28, 49.65, 53.96, 58.39, 60.7],
                     'Tip-Adapter-F': [49.47, 54.14, 57.74, 62.65, 66.61],
                     'APE': [52.66, 58.63, 60.7, 66.08, 67.73]},
           "zs": 42.85,
           "our": [50.11820330969267, 54.66903073286053, 59.33806146572104, 65.83924349881796, 71.63120567375887]}

    eurosat = {"other": {'CoOp': [51.3, 59.87, 70.3, 77.13, 83.27],
                         # 'LP-CLIP': [51.0, 61.58, 68.27, 76.93, 82.76],
                         'Proto-CLIP': [55.53, 64.89, 68.67, 69.42, 72.95],
                         'Proto-CLIP-F': [54.93, 64.86, 68.52, 78.94, 83.53],
                         'CLIP-Adapter': [61.10, 63.07, 73.14, 78.0, 84.97],
                         'SuS-X': [55.88, 61.54, 68.14, 68.85, 73.16],
                         'Tip-Adapter': [54.43, 61.43, 65.52, 67.96, 70.63],
                         'Tip-Adapter-F': [59.10, 66.19, 73.46, 78.04, 84.98],
                         'APE': [59.79, 62.86, 70.33, 74.51, 78.16]},
               "zs": 36.07,
               "our": [73.92592592592592, 74.64197530864197, 80.49382716049382, 87.8888888888889, 90.19753086419753]}

    fgvc = {"other": {'CoOp': [8.87, 18.63, 20.3, 27.33, 31.67],
                      # 'LP-CLIP': [12.89, 17.85, 23.57, 29.55, 36.39],
                      'Proto-CLIP': [19.59, 22.14, 23.25, 27.63, 29.67],
                      'Proto-CLIP-F': [19.5, 22.14, 23.31, 31.32, 37.56],
                      'CLIP-Adapter': [17.60, 20.05, 22.47, 27.37, 31.37],
                      'SuS-X': [20.1, 21.93, 22.92, 26.91, 30.12],
                      'Tip-Adapter': [18.81, 21.24, 22.14, 25.5, 29.94],
                      'Tip-Adapter-F': [20.67, 23.76, 26.16, 30.27, 35.13],
                      'APE': [20.85, 22.89, 24.39, 28.32, 31.26]},
            "zs": 16.95,
            "our": [18.69186918691869, 21.54215421542154, 26.432643264326433, 39.363936393639364, 53.46534653465347]}

    food101 = {"other": {'CoOp': [74.3, 71.77, 73.33, 71.93, 74.47],
                         # 'LP-CLIP': [30.13, 42.79, 55.15, 63.82, 70.17],
                         'Proto-CLIP': [77.36, 77.34, 77.58, 77.9, 78.11],
                         'Proto-CLIP-F': [77.34, 77.34, 77.58, 78.29, 79.09],
                         'CLIP-Adapter': [77.43, 77.77, 78.10, 78.53, 79.03],
                         'SuS-X': [77.38, 77.53, 77.5, 77.87, 77.93],
                         'Tip-Adapter': [77.42, 77.5, 77.52, 77.7, 77.87],
                         'Tip-Adapter-F': [77.54, 77.73, 78.14, 78.51, 79.28],
                         'APE': [77.59, 77.64, 77.59, 78.27, 78.5]},
               "zs": 77.37,
               "our": [77.36963696369637, 77.36963696369637, 77.8052805280528, 78.97689768976898, 79.28052805280528]}

    imagenet = {"other": {'CoOp': [57.57, 57.86, 59.63, 61.57, 62.97],
                          # 'LP-CLIP': [22.07, 31.95, 41.29, 49.55, 55.87],
                          'Proto-CLIP': [60.31, 60.64, 61.3, 62.12, 62.77],
                          'Proto-CLIP-F': [60.32, 60.64, 61.3, 63.92, 65.75],
                          'CLIP-Adapter': [61.2, 61.3, 61.85, 62.8, 63.7],
                          'SuS-X': [60.73, 61.03, 61.1, 61.57, 62.16],
                          'Tip-Adapter': [60.68, 60.94, 60.98, 61.45, 62.03],
                          'Tip-Adapter-F': [61.32, 61.69, 62.52, 64.00, 65.51],
                          'APE': [62.04, 62.34, 62.54, 62.79, 63.42]},
                "zs": 60.34,
                "our": [60.482, 61.254, 62.47200000000001, 64.44, 66.632]}

    oxford_flowers = {"other": {'CoOp': [69.07, 78.00, 85.80, 91.40, 94.73],
                                # 'LP-CLIP': [58.07, 73.35, 84.8, 92.0, 94.95],
                                'Proto-CLIP': [76.98, 83.39, 88.23, 92.08, 92.94],
                                'Proto-CLIP-F': [77.47, 83.52, 88.27, 93.79, 95.78],
                                'CLIP-Adapter': [73.9, 82.03, 87.13, 90.50, 94.10],
                                'SuS-X': [73.44, 79.41, 85.92, 88.59, 90.29],
                                'Tip-Adapter': [73.2, 79.01, 83.8, 87.9, 89.81],
                                'Tip-Adapter-F': [80.02, 83.19, 89.44, 91.72, 94.44],
                                'APE': [79.62, 83.68, 87.9, 91.11, 91.96]},
                      "zs": 66.02,
                      "our": [79.98375964271214, 86.47990255785626, 91.55501421031262, 93.74746244417376, 96.38652050345108]}

    oxford_pets = {"other": {'CoOp': [85.87, 82.03, 87.03, 85.4, 86.30],
                             # 'LP-CLIP': [30.14, 43.47, 56.35, 65.94, 76.42],
                             'Proto-CLIP': [86.1, 87.38, 87.19, 88.04, 88.61],
                             'Proto-CLIP-F': [85.72, 87.38, 86.95, 88.55, 89.62],
                             'CLIP-Adapter': [85.8, 86.13, 86.76, 87.93, 88.73],
                             'SuS-X': [85.41, 87.95, 88.28, 89.12, 89.86],
                             'Tip-Adapter': [86.02, 86.97, 86.48, 86.94, 88.53],
                             'Tip-Adapter-F': [86.40, 87.08, 87.24, 88.09, 89.64],
                             'APE': [86.26, 87.06, 87.31, 87.55, 88.88]},
                   "zs": 85.75,
                   "our": [87.2172254020169, 87.18997001907877, 88.06214227309894, 89.09784682474789, 90.54238212046879]}

    stanford_cars = {"other": {'CoOp': [56.23, 58.73, 62.90, 67.85, 73.33],
                               # 'LP-CLIP': [24.64, 36.53, 48.42, 60.82, 70.08],
                               'Proto-CLIP': [57.29, 60.01, 63.33, 64.93, 68.11],
                               'Proto-CLIP-F': [57.34, 60.04, 63.34, 70.35, 75.25],
                               'CLIP-Adapter': [57.03, 57.87, 58.97, 68.02, 74.1],
                               'SuS-X': [58.79, 60.27, 63.84, 65.81, 67.3],
                               'Tip-Adapter': [57.38, 58.55, 61.52, 63.08, 66.71],
                               'Tip-Adapter-F': [58.85, 60.79, 64.92, 69.39, 75.31],
                               'APE': [59.57, 61.39, 65.09, 66.71, 70.31]},
                     "zs": 55.81,
                     "our": [56.970526053973394, 59.76868548687975, 64.81780873025743, 74.38129585872404, 81.20880487501555]}

    sun397 = {"other": {'CoOp': [60.03, 60.00, 63.40, 65.73, 69.40],
                        # 'LP-CLIP': [32.8, 44.44, 54.59, 62.17, 67.15],
                        'Proto-CLIP': [60.81, 63.12, 65.51, 67.37, 68.09],
                        'Proto-CLIP-F': [60.83, 63.2, 65.57, 69.59, 71.94],
                        'CLIP-Adapter': [60.6, 62.3, 63.7, 67.42, 69.73],
                        'SuS-X': [61.69, 63.38, 64.94, 66.28, 68.],
                        'Tip-Adapter': [61.27, 62.72, 64.24, 65.59, 66.83],
                        'Tip-Adapter-F': [62.57, 63.77, 65.59, 69.19, 71.37],
                        'APE': [64.37, 65.93, 66.64, 68.37, 69.72]},
              "zs": 58.83,
              "our": [62.584382871536526, 64.54911838790932, 67.81863979848866, 70.63476070528966, 72.72040302267003]}

    ucf101 = {"other": {'CoOp': [62.67, 63.60, 69.4, 72.5, 76.07],
                        # 'LP-CLIP': [41.43, 53.55, 62.23, 69.64, 73.72],
                        'Proto-CLIP': [63.15, 67.46, 69.5, 71.08, 73.35],
                        'Proto-CLIP-F': [63.07, 67.49, 69.55, 74.81, 77.5],
                        'CLIP-Adapter': [62.77, 64.17, 69.10, 72.9, 76.30],
                        'SuS-X': [62.78, 66.45, 66.87, 68.88, 71.95],
                        'APE': [63.26, 65.72, 69.92, 71.74, 74.49],
                        'Tip-Adapter': [62.73, 64.76, 66.19, 68.46, 70.68],
                        'Tip-Adapter-F': [64.95, 66.38, 70.98, 74.65, 77.5]},
              "zs": 61.78,
              "our": [66.82527094898228, 69.8123182659265, 74.12106793550093, 77.02881311128735, 82.26275442770287]}

    average = {"other": {'CoOp': [],
                         # 'LP-CLIP': [],
                         'Proto-CLIP': [],
                         'Proto-CLIP-F': [],
                         'CLIP-Adapter': [],
                         'SuS-X': [],
                         'APE': [],
                         'Tip-Adapter': [],
                         'Tip-Adapter-F': []},
               "zs": 0,
               "our": []}

    def __init__(self):
        cls = DataAndMethod

        self.data_dict = {"Caltech101": cls.caltech101, 'DTD': cls.dtd, 'EuroSAT': cls.eurosat,
                          'FGVCAircraft': cls.fgvc, 'Food101': cls.food101, 'ImageNet': cls.imagenet,
                          'Flowers102': cls.oxford_flowers, 'OxfordPets': cls.oxford_pets,
                          'StanfordCars': cls.stanford_cars, 'SUN397': cls.sun397, 'UCF101': cls.ucf101,
                          'Average': cls.average}
        if len(self.data_dict["Average"]["our"]) == 0:
            our_result = [self.data_dict[key]["our"] for key in self.data_dict][:-1]
            our_average = [sum([one[i] for one in our_result]) / len(our_result) for i, two in enumerate(our_result[0])]
            self.data_dict["Average"]["our"] = our_average
            for method_key in self.data_dict["Average"]["other"]:
                method_result = [self.data_dict[key]["other"][method_key] for key in self.data_dict][:-1]
                method_result_average = [sum([one[i] for one in method_result]) / len(method_result) for i, two in
                                         enumerate(method_result[0])]
                self.data_dict["Average"]["other"][method_key] = method_result_average
                pass
            zs_avg = [self.data_dict[dataset_key]["zs"] for dataset_key in self.data_dict if "Average" not in dataset_key]
            self.data_dict["Average"]["zs"] = sum(zs_avg) / len(zs_avg)
            pass

        for key in self.data_dict:
            now_r = " & ".join([f"{one:.2f}" for one in self.data_dict[key]['our']])
            Tools.print(f"{key}: {now_r}")
            pass
        for method_key in self.data_dict["Average"]["other"]:
            now_r = " & ".join([f"{one:.2f}" for one in self.data_dict["Average"]["other"][method_key]])
            Tools.print(f'{method_key} {now_r}')
            pass
        Tools.print(f'zs {self.data_dict["Average"]["zs"]}')
        pass

    pass


class VisFigure5(object):

    def __init__(self, result_path="./result/vis"):
        self.result_path = Tools.new_dir(result_path)
        self.data_and_method = DataAndMethod()
        pass

    def draw_curve(self, data_dict_list, data_zs, data_our_list, dataset_name):
        plt.figure(figsize=(6, 5))
        ax = plt.axes()
        ax.set_facecolor(color=(244 / 255, 244 / 255, 244 / 255, 1))
        plt.grid(color=(255 / 255, 255 / 255, 255 / 255, 1))

        line_styles = ["-", "-."]
        colors = [(78 / 255, 141 / 255, 153 / 255, 1), (105 / 255, 101 / 255, 53 / 255, 1),
                  (121 / 255, 114 / 255, 161 / 255, 1), (186 / 255, 85 / 255, 211 / 255, 1),
                  (200 / 255, 105 / 255, 104 / 255, 1)]
        x = [1, 2, 4, 8, 16]
        for index, key in enumerate(data_dict_list):
            plt.plot(x, data_dict_list[key], color=colors[index % len(colors)], label=key.replace("_", "-"),
                     linestyle=line_styles[index // len(colors)],
                     linewidth=1.5, marker='v', markersize=5)
            pass
        plt.plot(x, data_our_list, color=(254 / 255, 1 / 255, 1 / 255, 1), label='LDC(Ours)',
                 linestyle='-', linewidth=1.5, marker='*', markersize=7)
        plt.scatter([0], [data_zs], label='Zero-shot CLIP', s=90, marker='*',
                    color=(254 / 255, 1 / 255, 1 / 255, 1))

        plt.legend(loc=4, bbox_to_anchor=(0.6, 0.05), ncol=1, prop={'family': 'Times New Roman', 'size': 12},
                   mode="expand", numpoints=0.5, borderaxespad=0., fontsize=10, edgecolor="white")  #

        plt.xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
        plt.xticks(fontproperties='Times New Roman', fontsize=17)  #
        plt.yticks(fontproperties='Times New Roman', fontsize=17)
        plt.ylabel('ACC (%)', fontdict={'family': 'Times New Roman', 'size': 18})
        plt.xlabel('Number of Shots', fontdict={'family': 'Times New Roman', 'size': 18})
        plt.xlim((0.2, 16.8))
        plt.ylim(self._get_min_and_max(data_dict_list, data_zs, data_our_list))
        plt.title('{}'.format(dataset_name), fontdict={'family': 'Times New Roman', 'size': 20})

        plt.tight_layout(pad=0.1, w_pad=1, h_pad=-0.1)
        plt.savefig(os.path.join(self.result_path, '{}.png'.format(dataset_name)))
        # plt.show()
        pass

    def runner(self):
        for dataset_name in self.data_and_method.data_dict:
            self.draw_curve(self.data_and_method.data_dict[dataset_name]["other"],
                            self.data_and_method.data_dict[dataset_name]["zs"],
                            self.data_and_method.data_dict[dataset_name]["our"], dataset_name)
        pass

    def draw_one(self, ax, data_dict_list, data_zs, data_our_list, dataset_name):
        ax.set_facecolor(color=(244 / 255, 244 / 255, 244 / 255, 1))
        ax.grid(color=(255 / 255, 255 / 255, 255 / 255, 1))

        line_styles = ["-", "-."]
        colors = [(78 / 255, 141 / 255, 153 / 255, 1), (105 / 255, 101 / 255, 53 / 255, 1),
                  (121 / 255, 114 / 255, 161 / 255, 1), (186 / 255, 85 / 255, 211 / 255, 1),
                  (200 / 255, 105 / 255, 104 / 255, 1)]
        x = [1, 2, 4, 8, 16]
        for index, key in enumerate(data_dict_list):
            ax.plot(x, data_dict_list[key], color=colors[index % len(colors)], label=key.replace("_", "-"),
                    linestyle=line_styles[index // len(colors)],
                    linewidth=1.5, marker='v', markersize=5)
            pass
        ax.plot(x, data_our_list, color=(254 / 255, 1 / 255, 1 / 255, 1), label='LDC(Ours)',
                linestyle='-', linewidth=1.5, marker='*', markersize=7)
        ax.scatter([0], [data_zs], label='Zero-shot CLIP', s=90, marker='*',
                   color=(1 / 255, 1 / 255, 254 / 255, 1))

        ax.legend(loc=4, bbox_to_anchor=(0.6, 0.02), ncol=1, prop={'family': 'Times New Roman', 'size': 12},
                  mode="expand", numpoints=0.5, borderaxespad=0., fontsize=10, edgecolor="white")  #

        ax.set_xticks([0, 1, 2, 4, 8, 16], [0, 1, 2, 4, 8, 16], fontproperties='Times New Roman', fontsize=17)
        ax.set_xlim((-0.8, 16.8))
        y_ticks = [int(one) for one in ax.get_yticks()]
        ax.set_ylim(self._get_min_and_max(data_dict_list, data_zs, data_our_list))
        ax.set_yticks(y_ticks, y_ticks, fontproperties='Times New Roman', fontsize=17)
        ax.set_ylabel('ACC (%)', fontdict={'family': 'Times New Roman', 'size': 18})
        ax.set_xlabel('Number of Shots', fontdict={'family': 'Times New Roman', 'size': 18})
        ax.set_title('{}'.format(dataset_name), fontdict={'family': 'Times New Roman', 'size': 20})
        pass

    def runner_together(self):
        dataset_name_list = [["Caltech101", 'DTD', 'EuroSAT', 'FGVCAircraft'],
                             ['Food101', 'Flowers102', 'OxfordPets', 'StanfordCars'],
                             ['SUN397', 'UCF101', 'ImageNet', 'Average']]
        row = len(dataset_name_list)
        col = len(dataset_name_list[0])
        fig, axs = plt.subplots(row, col, figsize=(col * 6, row * 6))
        for i in range(row):
            for j in range(col):
                dataset_name = dataset_name_list[i][j]
                self.draw_one(axs[i, j], self.data_and_method.data_dict[dataset_name]["other"],
                              self.data_and_method.data_dict[dataset_name]["zs"],
                              self.data_and_method.data_dict[dataset_name]["our"], dataset_name)
                pass
            pass
        plt.tight_layout(pad=1, w_pad=2, h_pad=2)
        plt.savefig(os.path.join(self.result_path, 'Figure5.pdf'))
        # plt.show()
        pass

    @staticmethod
    def _get_min_and_max(data_dict_list, data_zs, data_our_list):
        all_num = [data_dict_list[key] for key in data_dict_list] + [data_our_list]
        y_min = min([min(min(all_num)), data_zs])
        y_max = max([max(max(all_num)), data_zs])
        return (int(y_min), int(y_max + 1))

    pass


if __name__ == '__main__':
    # Read result
    read_result = ReadResult(txt_path="./result/log/1_main_experiment_1_zero_shot.txt")
    read_result.get_result()
    read_result = ReadResult(txt_path="./result/log/1_main_experiment_2_few_shot.txt")
    read_result.get_result()

    # Vis Figure 5
    vis_figure5 = VisFigure5(result_path="./result/vis")
    vis_figure5.runner_together()
    pass
