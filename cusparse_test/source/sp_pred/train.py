import torch
import numpy as np
import copy
from pred import *


def gen_mask(x, p=0):
    if p == 0:
        return (x>0).to(torch.float32)
    max_abs_thres = x.abs().max(dim=2, keepdim=True).values*p
    return (x>max_abs_thres).to(torch.float32)


def evaluate(model, device, x, y):
    model.eval()
    eval = {
        "Loss": [],
        "Loss Weight": [],
        "Recall": [],
        "Classifier Sparsity": [],
        "True Sparsity": [],
    }

    with torch.no_grad():
        for i in range(int(x.size(0)/1000)):
            xb = x[i*1000:(i+1)*1000].to(torch.float32).to(device)
            yb = y[i*1000:(i+1)*1000].to(torch.float32).to(device)
            logits = model(xb)
            preds = logits >= 0.5

            dif = yb.int() - preds.int()
            miss = dif > 0.0  # classifier didn't activated target neuron

            weight = (yb.sum() / yb.numel()) + 0.005
            loss_weight = yb * (1 - weight) + weight
            eval["Loss Weight"] += [weight.item()]
            eval["Loss"] += [
                torch.nn.functional.binary_cross_entropy(
                    logits, yb, weight=loss_weight
                ).item()
            ]

            eval["Recall"] += [((yb.sum(dim=1).float() - miss.sum(dim=1).float()).mean().item())]
            eval["True Sparsity"] += [yb.sum(dim=1).float().mean().item()]
            eval["Classifier Sparsity"] += [preds.sum(dim=1).float().mean().item()]

    for k, v in eval.items():
        eval[k] = np.array(v).mean()

    eval["Recall"] = eval["Recall"] / eval["True Sparsity"]
    return eval


def eval_print(validation_results):
    result = ""
    for metric_name, metirc_val in validation_results.items():
        result = f"{result}{metric_name}: {metirc_val:.4f} "
    return result


def train_one_layer(model, x, y, xv, yv, args):
    device = args["device"]
    best_model = copy.deepcopy(model.state_dict())
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), args['lr'], weight_decay=args['wd'])
    no_improve = 0
    base_acc = 0
    for e in range(args['epoch']):
        for i in range(int(45000/args['bs'])):
            model.train()
            optimizer.zero_grad()
            xb = x[i*args['bs']:(i+1)*args['bs']].to(torch.float32).to(device)
            yb = y[i*args['bs']:(i+1)*args['bs']].to(torch.float32).to(device)
            logits = model(xb)

            weight = (yb.sum() / yb.numel()) + 0.005
            loss_weight = yb * (1 - weight) + weight
            loss = torch.nn.functional.binary_cross_entropy(logits, yb, weight=loss_weight)
            loss.backward()
            optimizer.step()

            # if (i + 1) % 5 == 0:
            #     print(f"[{e}, {i}] Loss: {loss.item():.4f}, Loss weight: {weight.item():.4f}")

            # if ((i + 1) % 5 == 0):
            #     valid_eval_results = evaluate(model, device, xv, yv)
            #     train_eval_results = evaluate(model, device, xb, yb)
            #     model.train()
            #     print(f"[{e}, {i}] Validation: {eval_print(valid_eval_results)}")
            #     print(f"[{e}, {i}] Train: {eval_print(train_eval_results)}")

        train_eval_results = evaluate(model, device, x, y)
        epoch_eval_results = evaluate(model, device, xv, yv)
        print(f"[Epoch {e+1}] [Train] {eval_print(train_eval_results)}")
        print(f"[Epoch {e+1}] [Valid] {eval_print(epoch_eval_results)}", flush=True)

        if epoch_eval_results["Recall"] > base_acc:
            base_acc = epoch_eval_results["Recall"]
            best_eval = epoch_eval_results
            model.cpu()
            best_model = copy.deepcopy(model.state_dict())
            model.to(device)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= 5 or base_acc > 0.99:
            break

    # model.cpu()
    # model.load_state_dict(best_model)
    return best_model, best_eval


if __name__ == "__main__":
    prenorm = torch.load("/share/public/zhouyongkang/projects/sc/data/gen_data/prenorm.pt",
                         weights_only=True)
    inter_result = torch.load(
        "/share/public/zhouyongkang/projects/sc/data/gen_data/inter_result.pt",
        weights_only=True
    )
    mask = gen_mask(inter_result)

    args = {
        "lr": 5e-4,
        "wd": 1e-2,
        "epoch": 2000,
        "bs": 1000,
        "device": 'cuda:0',
    }

    models = [Predictor(4096, 14336) for _ in range(32)]

    for i in range(32):
        x = prenorm[i, :45000]
        y = mask[i, :45000]
        xv = prenorm[i, 45000:]
        yv = mask[i, 45000:]
        print(f"\n\nTrain pred of layer{i}\n")
        train_one_layer(models[i], x, y, xv, yv, args)
    
    torch.save(models, "/share/xujiaming/train_machine/yongkang/cusparse_test/data/preds/pred.pt")
