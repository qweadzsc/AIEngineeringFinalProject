from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


torch.set_printoptions(precision=6)
model_path = "/share/public/zhouyongkang/models/"
model_name = "TurboSparse-7B-mdf"

layer = 10
device = torch.device('cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_path+model_name,
                                             torch_dtype=torch.float16,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path+model_name)
predictor = model.model.layers[layer].mlp.predictor
predictor = predictor.to(torch.float32).to(device)

prenorm_data = torch.load(
    "/share/public/zhouyongkang/projects/sc/data/gen_data/prenorm_mmlu.pt",
    weights_only=True)[layer]
inter_data = torch.load(
    "/share/public/zhouyongkang/projects/sc/data/gen_data/inter_result_mmlu.pt",
    weights_only=True)[layer]
x = prenorm_data.to(torch.float32).to(device)
label = (inter_data != 0).to(torch.float32).to(device)

log_path="/share/public/zhouyongkang/projects/sc/src/pred/train.log"

# ----------------------------------------------------------------

# 手动划分训练集和测试集，80% 训练，20% 测试
train_size = int(0.95 * len(x))
X_train = x[:train_size]
y_train = label[:train_size]
X_test = x[train_size:]
y_test = label[train_size:]

# 初始化参数
num_epochs = 100000
learning_rate = 0.001
optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)
# 为每个输出维度的正/负值设置初始权重，正例关注度高
pos_weight = torch.full((14336,), 0.98, dtype=torch.float32)
neg_weight = torch.full((14336,), 0.02, dtype=torch.float32)
weights = torch.stack([neg_weight, pos_weight], dim=1).to(device)

def weighted_binary_cross_entropy(output, target):
    output = torch.clamp(output, min=1e-7, max=1 - 1e-7)
    loss = target * torch.log(output) * weights[:, 1] + (1 - target) * torch.log(1 - output) * weights[:, 0]
    return -torch.mean(loss)

criterion = weighted_binary_cross_entropy
no_decrease_count = 0
last_recall = 0

# 打开 log 文件
log_file = open(log_path, 'a+')

for epoch in range(num_epochs):
    predictor.train()
    optimizer.zero_grad()
    outputs = torch.sigmoid(predictor(X_train))
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # 每个 epoch 计算查全率
    predictor.eval()
    with torch.no_grad():
        test_outputs = torch.sigmoid(predictor(X_test))
        predicted = (test_outputs > 0.5).float()

        true_positives = ((predicted == 1) & (y_test == 1)).sum(dim=0).float()
        false_negatives = ((predicted == 0) & (y_test == 1)).sum(dim=0).float()
        recall_per_dim = true_positives / (true_positives + false_negatives)
        recall_per_dim[torch.isnan(recall_per_dim)] = 1.0
        recall = recall_per_dim.mean().item()

        true_positives_pred = ((predicted == 1) & (y_test == 1)).sum(dim=0).float()
        false_positives = ((predicted == 1) & (y_test == 0)).sum(dim=0).float()
        precision_per_dim = true_positives_pred / (true_positives_pred + false_positives)
        precision_per_dim[torch.isnan(precision_per_dim)] = 1.0
        precision = precision_per_dim.mean().item()

        posterior_activation_rate = (y_test == 1).sum(dim=0).float() / y_test.size(0)
        posterior_activation_rate = posterior_activation_rate.mean().item()

        prediction_activation_rate = (predicted == 1).sum(dim=0).float() / predicted.size(0)
        prediction_activation_rate = prediction_activation_rate.mean().item()

    # 检查查全率是否降低
    if recall >= last_recall:
        no_decrease_count += 1
        if no_decrease_count == 5:
            # 提高负例的权重
            weights[:, 0] *= 1.15
            no_decrease_count = 0
    else:
        no_decrease_count = 0
    last_recall = recall
    if last_recall < 0.8:
        break

    # 每五个 epoch 记录日志
    if (epoch + 1) % 5 == 0:
        log_msg = f'Epoch {epoch + 1}: Recall={recall}, Precision={precision}, ' \
                    f'Posterior Activation Rate={posterior_activation_rate}, ' \
                    f'Prediction Activation Rate={prediction_activation_rate}\n'
        log_file.write(log_msg)
        log_file.flush()

# 关闭 log 文件
log_file.close()