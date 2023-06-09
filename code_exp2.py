import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from PIL import Image
from random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchmetrics import F1Score
from transformers import BertTokenizer, BertModel

class BertUnet(nn.Module):
    def __init__(self, model_name, output_dim):
        super(BertUnet, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.unet = Unet()
        self.mlp = nn.Linear(768, output_dim)# 768 for small, 1024 for large

    #def forward(self, text_input):
    def forward(self, text_input, image_input):    
        text_output = self.bert(**text_input)
        image_output = self.unet(image_input)
        image_output = image_output.view(-1, 32 * 32)
        image_output = image_output.narrow(1, 0, 768)# clip
        #output = self.mlp(text_output[1])
        output = self.mlp(text_output[1] * image_output)
        
        return output

class cov(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(cov, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace = True)
            )

    def forward(self, input):

        return self.layer(input)

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.cov_down1 = cov(3, 32)
        self.cov_down2 = cov(32, 64)
        self.cov_down3 = cov(64, 128)
        self.cov_down4 = cov(128, 256)
        self.cov = cov(256, 512)
        self.down = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.up4 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.up3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.up2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.up1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.cov_up4 = cov(256 + 256, 256)
        self.cov_up3 = cov(128 + 128, 128)
        self.cov_up2 = cov(64 + 64, 64)
        self.cov_up1 = cov(32 + 32, 32)
        self.cov_out = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 3, padding = 1), 
            nn.BatchNorm2d(1), 
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
            )

    def forward(self, input):
        output1 = self.cov_down1(input)
        output = self.down(output1)
        output2 = self.cov_down2(output)
        output = self.down(output2)
        output3 = self.cov_down3(output)
        output = self.down(output3)
        output4 = self.cov_down4(output)
        output = self.down(output4)
        output = self.cov(output)
        output = self.up4(output)
        output = torch.cat([output, output4], dim = 1)
        output = self.cov_up4(output)
        output = self.up3(output)
        output = torch.cat([output, output3], dim = 1)
        output = self.cov_up3(output)
        output = self.up2(output)
        output = torch.cat([output, output2], dim = 1)
        output = self.cov_up2(output)
        output = self.up1(output)
        output = torch.cat([output, output1], dim = 1)
        output = self.cov_up1(output)
        output = self.cov_out(output)

        return output

def train_valid_split(dataset, valid_size, shuffle = True, random_state = 2023):
    if shuffle:
        dataset = dataset.sample(frac = 1.0, random_state = random_state).reset_index(drop = True)
    trainset = dataset[(int)(len(dataset) * valid_size):].reset_index(drop = True)
    validset = dataset[:(int)(len(dataset) * valid_size)].reset_index(drop = True)
    
    return trainset, validset

def evaluate(model, dataset, batch_size, device):
    batch_num = len(dataset["情感倾向"]) // batch_size
    prediction = []
    for batch in range(batch_num):
        # text input
        text_input = dataset["微博中文内容"][(batch_size * batch) : (batch_size * (batch + 1))].to_list()
        text_input = tokenizer(text_input, padding = True, truncation = True, max_length = 140, return_tensors = "pt").to(device)
        # image input
        
        image_input = torch.Tensor()
        blank_pic = torch.zeros(1, 3, 128, 128)
        for image_str in dataset["微博图片"][(batch_size * batch) : (batch_size * (batch + 1))]:
            if image_str == "":
                image_input = torch.cat([image_input, blank_pic], dim = 0)
            else:
                image_path_list = image_str.split(", ")
                try:
                    image_path = choice(image_path_list)# pick a random image
                    image = Image.open(requests.get(image_path, verify = False, stream = True).raw).convert('RGB')
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128), interpolation = 2)])# C, H, W
                    image = transform(image)
                    image = image.unsqueeze(0)
                    image_input = torch.cat([image_input, image], dim = 0)
                except:
                    image_input = torch.cat([image_input, blank_pic], dim = 0)
        image_input = image_input.to(device)
        
        with torch.no_grad():
            #output = model(text_input)
            output = model(text_input, image_input)
            output = F.softmax(output, dim = -1)
            prediction.extend(output.argmax(dim = -1).cpu().tolist())
    labels = torch.tensor(dataset["情感倾向"].to_list()) + 1
    prediction = torch.tensor(prediction)
    f1score = F1Score(task = "multiclass", num_classes = 3, average = "macro")
    f1 = f1score(prediction, labels)
    cnt = (torch.tensor(labels) == torch.tensor(prediction)).sum()
    acc = cnt / len(dataset["情感倾向"])
    
    return f1, acc

def train(model, dataset, device):
    epoch_num = 10 # modify
    batch_size = 30 # modify according to gpu
    lr = 1e-4 # modify
    loss_function = nn.CrossEntropyLoss()
    trainset, validset = train_valid_split(dataset, valid_size = 0.1, shuffle = True, random_state = 2023)
    batch_num = len(trainset["情感倾向"]) // batch_size
    optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    # before training
    train_f1, train_acc = evaluate(model, trainset, batch_size, device)
    valid_f1, valid_acc = evaluate(model, validset, batch_size, device)
    print("----Before training----")
    print("train_f1 = {:.4f}, train_acc = {:.4f}, valid_f1 = {:.4f}, valid_acc = {:.4f}".format(train_f1, train_acc, valid_f1, valid_acc))
    # train
    for epoch in range(epoch_num):
        loss_sum = 0
        for batch in range(batch_num):
            # text input
            text_input = trainset["微博中文内容"][(batch_size * batch) : (batch_size * (batch + 1))].to_list()
            text_input = tokenizer(text_input, padding = True, truncation = True, max_length = 140, return_tensors = "pt").to(device)
            # image input
            
            image_input = torch.Tensor()
            blank_pic = torch.zeros(1, 3, 128, 128)
            for image_str in trainset["微博图片"][(batch_size * batch) : (batch_size * (batch + 1))]:
                if image_str == "":
                    image_input = torch.cat([image_input, blank_pic], dim = 0)
                else:
                    try:
                        image_path_list = image_str.split(", ")
                        image_path = choice(image_path_list)# pick a random image
                        image = Image.open(requests.get(image_path, verify = False, stream = True).raw).convert('RGB')
                        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128), interpolation = 2)])# C, H, W
                        image = transform(image)
                        image = image.unsqueeze(0)
                        image_input = torch.cat([image_input, image], dim = 0)
                    except:
                        image_input = torch.cat([image_input, blank_pic], dim = 0)
            image_input = image_input.to(device)
            
            #output = model(text_input)
            output = model(text_input, image_input)
            labels = (torch.tensor(trainset["情感倾向"][(batch_size * batch) : (batch_size * (batch + 1))].to_list()) + 1).to(device)
            optimizer.zero_grad()
            l = loss_function(output, labels)
            l.backward()
            optimizer.step()
            loss_sum += l.detach()
        train_f1, train_acc = evaluate(model, trainset, batch_size, device)
        valid_f1, valid_acc = evaluate(model, validset, batch_size, device)
        loss = loss_sum / len(trainset["情感倾向"])
        print("----Epoch = {}----".format(epoch))
        print("loss = {:.4f}, train_f1 = {:.4f}, train_acc = {:.4f}, valid_f1 = {:.4f}, valid_acc = {:.4f}".format(loss, train_f1, train_acc, valid_f1, valid_acc))

def test(model, dataset, device):
    batch_size = 20 # modify according to gpu
    batch_num = len(dataset["微博id"]) // batch_size
    prediction = []
    for batch in range(batch_num):
        # text input
        text_input = dataset["微博中文内容"][(batch_size * batch) : (batch_size * (batch + 1))].to_list()
        text_input = tokenizer(text_input, padding = True, truncation = True, max_length = 140, return_tensors = "pt").to(device)
        # image input
        
        image_input = torch.Tensor()
        blank_pic = torch.zeros(1, 3, 128, 128)
        for image_str in dataset["微博图片"][(batch_size * batch) : (batch_size * (batch + 1))]:
            if image_str == "":
                image_input = torch.cat([image_input, blank_pic], dim = 0)
            else:
                try:
                    image_path_list = image_str.split(", ")# pick a random image
                    image_path = choice(image_path_list)
                    image = Image.open(requests.get(image_path, verify = False, stream = True).raw).convert('RGB')
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128), interpolation = 2)])# C, H, W
                    image = transform(image)
                    image = image.unsqueeze(0)
                    image_input = torch.cat([image_input, image], dim = 0)
                except:
                    image_input = torch.cat([image_input, blank_pic], dim = 0)
        image_input = image_input.to(device)
        
        with torch.no_grad():
            #output = model(text_input)
            output = model(text_input, image_input)
            output = F.softmax(output, dim = -1)
            prediction.extend((output.argmax(dim = -1) - 1).cpu().tolist())
    if len(dataset) == 10000:
        submission = pd.DataFrame({"index": dataset["微博id"], "y": prediction})
        submission = submission.rename(columns = {"微博id":"id"})
        submission.to_csv("submission_bert_unlabeled_image.csv", index = False)
    if len(dataset) == 900000:
        dataset["情感倾向"] = prediction

def deal_dataset(dataset):
    dataset = dataset.drop(columns = "发布人账号")
    dataset["微博中文内容"] = dataset["微博中文内容"].astype(str)
    #dataset["微博中文内容"].str.cat(dataset["微博发布时间"].astype(str))
    dataset["微博发布时间"] = dataset["微博发布时间"].apply(lambda x: "2020-" + x.replace("月", "-").replace("日", "").split(" ")[0])
    dataset["微博发布时间"] = pd.to_datetime(dataset["微博发布时间"], format = "%Y-%m-%d")
    dataset["微博图片"] = dataset["微博图片"].astype(str)
    dataset["微博图片"] = dataset["微博图片"].apply(lambda x: x.replace("[]", "").replace("[", "").replace("]", "").replace("'", ""))
    # clear
    
    dataset["微博中文内容"] = dataset["微博中文内容"].apply(lambda x: x.replace("转发微博", "").replace("?展开全文c", ""))
    dataset["微博中文内容"].replace("[0]?网页链接[\?]?", "", regex = True, inplace = True)
    dataset["微博中文内容"].replace("(http(s)?://)?[a-zA-Z0-9/&=%\_\.\?]+", "", regex = True, inplace = True)
    dataset["微博中文内容"].replace("((回复)?(//)?@[\w·]+:)+", "", regex = True, inplace = True)
    
    dataset["文本长度"] = dataset["微博中文内容"].str.len()
    dataset["图片长度"] = dataset["微博图片"].str.len()

    return dataset

# load dataset
trainset_labeled = pd.read_csv("nCoV_100k_train.labled.csv")
trainset_unlabeled = pd.read_csv("nCoV_900k_train.unlabled.csv")
testset = pd.read_csv("nCov_10k_test.csv")

# check information
#print(trainset_labeled.info())
#print(trainset_unlabeled.info())
#print(testset.info())

# deal with dataset
trainset_labeled = deal_dataset(trainset_labeled)
trainset_unlabeled = deal_dataset(trainset_unlabeled)
testset = deal_dataset(testset)

# remove irrelevant data
trainset_labeled = trainset_labeled[trainset_labeled["情感倾向"].isin(["-1", "0", "1"])].reset_index(drop = True)
trainset_labeled["情感倾向"] = trainset_labeled["情感倾向"].astype(int)
#print(trainset_labeled.info())

"""
# check the distribution of data
label_distribution_labeled = trainset_labeled.groupby("情感倾向", as_index = False).count()
plt.figure()
sns.barplot(x = "情感倾向", y = "微博id", data = label_distribution_labeled)
plt.xlabel("class")
plt.ylabel("occurance")
plt.savefig("label_distribution_labeled.png")
date_label_distribution_labeled = trainset_labeled.groupby(["微博发布时间", "情感倾向"], as_index = False).count()
plt.figure()
sns.lineplot(x = "微博发布时间", y = "微博id", hue = "情感倾向", data = date_label_distribution_labeled)
plt.xlabel("date")
plt.ylabel("occurance")
plt.legend([-1, 0, 1])
plt.savefig("date_label_distribution_labeled.png")
date_distribution_test = testset.groupby("微博发布时间", as_index = False).count()
plt.figure()
sns.lineplot(x = "微博发布时间", y = "微博id", data = date_distribution_test)
plt.xlabel("date")
plt.ylabel("occurance")
plt.savefig("date_distribution_test.png")
date_distribution_unlabeled = trainset_unlabeled.groupby("微博发布时间", as_index = False).count()
plt.figure()
sns.lineplot(x = "微博发布时间", y = "微博id", data = date_distribution_unlabeled)
plt.xlabel("date")
plt.ylabel("occurance")
plt.savefig("date_distribution_unlabeled.png")
text_length_distribution_labeled = trainset_labeled.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = text_length_distribution_labeled).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("text_length_distribution_labeled.png")
text_length_distribution_test = testset.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = text_length_distribution_test).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("text_length_distribution_test.png")
text_length_distribution_unlabeled = trainset_unlabeled.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = text_length_distribution_unlabeled).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("text_length_distribution_unlabeled.png")
image_length_distribution_labeled = trainset_labeled.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = image_length_distribution_labeled).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("image_length_distribution_labeled.png")
image_length_distribution_test = testset.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = image_length_distribution_test).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("image_length_distribution_test.png")
image_length_distribution_unlabeled = trainset_unlabeled.groupby("文本长度", as_index = False).count()
plt.figure()
sns.barplot(x = "文本长度", y = "微博id", data = image_length_distribution_unlabeled).set_xticks(ticks = list(range(0, 180, 10)))
plt.xlabel("length")
plt.ylabel("occurance")
plt.savefig("image_length_distribution_unlabeled.png")
print("----process over----")
"""
# load pre-trained model
device = torch.device("cuda:7")
model_name = "hfl/chinese-bert-wwm-ext"
model = BertUnet(model_name, output_dim = 3).to(device)
tokenizer = BertTokenizer.from_pretrained(model_name)
# train and test
train(model = model, dataset = trainset_labeled[0:99900], device = device)
test(model = model, dataset = trainset_unlabeled, device = device)
train(model = model, dataset = trainset_unlabeled, device = device)
test(model = model, dataset = testset, device = device)