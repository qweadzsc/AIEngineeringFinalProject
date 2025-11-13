import re
import matplotlib.pyplot as plt


def plot_log_data(file_path):
    recalls = []
    prediction_activation_rates = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 使用正则表达式提取所需数据
                match = re.search(r'Epoch (\d+): Recall=([\d.]+),.*Prediction Activation Rate=([\d.]+)', line)
                if match:
                    recall = float(match.group(2))
                    prediction_activation_rate = float(match.group(3))

                    recalls.append(recall)
                    prediction_activation_rates.append(prediction_activation_rate)

        # 绘制折线图
        plt.figure(figsize=(10, 6))
        plt.scatter(recalls, prediction_activation_rates)
        plt.ylim(0.65, 1)

        plt.xlabel('Recall')
        plt.ylabel('Prediction Activation Rate')
        plt.title('Prediction Activation Rate vs Recall')
        plt.grid(True)
        plt.savefig("./aaa.png")

    except FileNotFoundError:
        print(f"错误：未找到文件 {file_path}。")
    except Exception as e:
        print(f"发生未知错误：{e}")


# 调用函数并传入文件路径
file_path = './train.log'
plot_log_data(file_path)
