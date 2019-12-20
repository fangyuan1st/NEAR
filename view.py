import re
import matplotlib.pyplot as plt

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
            res = re.search(r'- loss in epoch (\d*) : (-?[\d\.]*)[^\d]', line)
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
                predict_f1.append(float(re.search(r'\'f1\': ([\d\.]*)[^\d]', line).group(1)))
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

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(loss_x, loss_loss, label='train_loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(valid_x, valid_loss, label='valid_loss')
plt.legend()

plt.subplot(2, 2, 3)
if len(predict_x):
    plt.plot(predict_x, predict_f1, label='f1')
    plt.plot(predict_x, predict_auc, label='auc')
    plt.plot(predict_x, predict_accuracy, label='accuracy')
else:
    plt.plot(node_x, node_macro_f1, label='macro_f1')
    plt.plot(node_x, node_micro_f1, label='micro_f1')
    plt.plot(node_x, node_accuracy, label='accuracy')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(noise_x, noise_origin_mean, label='origin_mean')
plt.plot(noise_x, noise_noise_mean, label='noise_mean')
plt.legend()
plt.show()
