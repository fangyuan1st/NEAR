import re

loss_x = []
loss_loss = []
valid_x = []
valid_loss = []
grid_x = []
grid_loss = []
predict_x = []
predict_f1 = []
predict_auc = []
predict_accuracy = []
node_x = []
node_micro_f1 = []
node_accuracy = []
node_macro_f1 = []
noise_x = []
noise_origin_mean = []
noise_noise_mean = []
with open(input(), 'r') as fi:
    for line in fi:
        if re.search(r'epoch', line):
            res = re.search(r'loss in epoch (\d*) : (-?[\d\.]*)[^\d]', line)
            if res:
                loss_x.append(int(res.group(1)))
                loss_loss.append(float(res.group(2)))
            res = re.search(r'evaluation loss in epoch (\d*) : (-?[\d\.]*)[^\d]', line)
            if res:
                valid_x.append(int(res.group(1)))
                valid_loss.append(float(res.group(2)))
            res = re.search(r'grid search loss in epoch (\d*) : (-?[\d\.]*)[^\d]', line)
            if res:
                grid_x.append(int(res.group(1)))
                grid_loss.append(float(res.group(2)))
            res = re.search(r'link prediction prediction epoch (\d*):', line)
            if res:
                predict_x.append(int(res.group(1)))
                predict_f1.append(float(re.search(r'\'micro_f1\': ([\d\.]*)[^\d]', line).group(1)))
                predict_auc.append(float(re.search(r'\'auc_roc\': ([\d\.]*)[^\d]', line).group(1)))
                predict_accuracy.append(float(re.search(r'\'accuracy\': ([\d\.]*)[^\d]', line).group(1)))
            res = re.search(r'node classification classification epoch (\d*):', line)
            if res:
                node_x.append(int(res.group(1)))
                node_micro_f1.append(float(re.search(r'\'micro_f1\': ([\d\.]*)[^\d]', line).group(1)))
                node_accuracy.append(float(re.search(r'\'accuracy\': ([\d\.]*)[^\d]', line).group(1)))
                node_macro_f1.append(float(re.search(r'\'macro_f1\': ([\d\.]*)[^\d]', line).group(1)))
            res = re.search(r'noise filter evaluation epoch (\d*):',line)
            if res:
                noise_x.append(int(res.group(1)))
                noise_origin_mean.append(float(re.search(r'\'original_mean\': ([\d\.]*)[^\d]',line).group(1)))
                if re.search(r'\'noise_mean\': ([\d\.]*)[^\d]',line).group(1) == '':
                    noise_noise_mean.append(0.0)
                    continue
                noise_noise_mean.append(float(re.search(r'\'noise_mean\': ([\d\.]*)[^\d]',line).group(1)))

k = valid_loss.index(min(valid_loss))
epoch = valid_x[k]
print('epoch: %d'%epoch)

grid_k = grid_x.index(epoch)
res = {'grid_loss':grid_loss[grid_k]}
print(res)

if epoch in predict_x:
    predict_k = predict_x.index(epoch)
    res = {'f1':predict_f1[predict_k],
           'auc':predict_auc[predict_k],
           'accuracy':predict_accuracy[predict_k]}
    print(res)

if epoch in node_x:
    node_k = node_x.index(epoch)
    res = {'micro_f1':node_micro_f1[node_k],
       'accuracy':node_accuracy[node_k],
       'macro_f1':node_macro_f1[node_k]}
    print(res)

if epoch in noise_x:
    noise_k = noise_x.index(epoch)
    res = {'origin_mean':noise_origin_mean[noise_k],
           'noise_mean': noise_noise_mean[noise_k]}
    print(res)
