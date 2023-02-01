from django.shortcuts import render,redirect
from django.contrib.auth import authenticate,login,logout
from django.contrib.auth.models import User
from django.contrib import messages
from .DrawRectangle.DrawBoundingBox import *

from .CNN.CNN import *
# from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import cv2


import os
import requests
cfp=os.path.abspath(os.path.dirname(__file__))
# Create your views here.
def loginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')
        try:
            user=User.objects.get(username=username)
        except:
            messages.error(request,'User does not exist')
        user=authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            messages.error(request,'Username OR password does not exist')
    return render(request,'base/login_register.html')

def logoutUser(request):
    logout(request)
    return redirect('home')


def home(request):
    return render(request,'base/home.html')

def draw_rectangles(request):
    return render(request,'base/draw_rectangles.html')

def draw(request):
    print(os.getcwd())

    # img_loc = cfp+'/DrawRectangle/test0.jpg'
    img_loc = cfp + '/DrawRectangle/demo_img.png'
    draw_rects = DrawRects(img_loc, (0, 255, 0), 1)
    cv2.namedWindow(WIN_NAME, 0)
    cv2.setMouseCallback(WIN_NAME, onmouse_draw_rect, draw_rects)
    while True:
        cv2.imshow(WIN_NAME,draw_rects.image_for_show)
        key = cv2.waitKey(30)
        if key == 27:  # ESC
            cv2.imwrite(cfp+'/DrawRectangle/Img_Boxes/'+draw_rects.img_name+'_bboxes'+'.jpg', draw_rects.image_for_show)
            break
    cv2.destroyAllWindows()

    return render(request,'base/draw_rectangles.html')

# def CNN_train(request):
#     train_dataset = RetrievalDataset("../base/DrawRectangle/base_imgs", "train")
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
#     s = iter(train_dataloader)
#     x = s.next()
#     train_dataset = RetrievalDataset(root, "train")
#     eval_dateset = RetrievalDataset(root, "eval")
#     test_dateset = RetrievalDataset(root, "test")
#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
#     eval_dataloader = DataLoader(eval_dateset, batch_size=64, shuffle=True, drop_last=True)
#     test_dataloader = DataLoader(test_dateset, batch_size=64, shuffle=True, drop_last=True)
#
#     num_epoch = 30
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     fypCNN = CNN()
#     fypCNN = fypCNN.to(device)
#     max_accuracy = 0
#     learning_rate = 1e-5  ##
#     optimizer = torch.optim.Adam(fypCNN.parameters(), lr=learning_rate)
#     iter_num = 0
#     for epoch in range(num_epoch):  # loop over the dataset multiple times
#         ## Train phase
#         fypCNN.train()
#         train_loss_epoch = 0
#         for inputs, labels in train_dataloader:
#             running_loss_iter = 0.0
#             iter_num += 1
#             inputs = inputs.to(device)
#             out = fypCNN(inputs)  ## step 1.
#             loss = loss_fn(out, labels)  ## step 2.
#             optimizer.zero_grad()  ## clear the previous gradients
#             loss.backward()  ## step 3. backpropagation - compute gradient
#             optimizer.step()  ## step 4. w = w - eta*w.grad
#             running_loss_iter += loss.item()
#             train_loss_iter = running_loss_iter / len(inputs)
#             # writer.add_scalar('training_loss_per_iter_triplet',train_loss_iter,iter_num)
#             if (iter_num % 100 == 0):
#                 print('train_loss_iter:', train_loss_iter)
#
#         # validation phase
#         if iter_num % 1000 == 0:
#             with torch.no_grad():
#                 fypCNN.eval()
#                 correct = 0
#                 for inputs, labels in eval_dataloader:
#                     inputs, labels = inputs.to(device), labels.cuda()
#                     out = fypCNN(inputs)
#                     accuracy = checkAccuracy(out, labels)  ##TODO: write checkAccuracy function
#                 if accuracy > max_accuracy:
#                     torch.save(fypCNN.state_dict(), 'net_parameters_binary_classification.pth')
#                     max_accuracy = accuracy
#                 # writer.add_scalar('validation_accuracy',accuracy,iter_num)
#                 print('iter:', iter_num, ' accuracy:', accuracy)
#                 fypCNN.train()

def preprocess(img):
    """
    It may be helpful to extract this preprocessing function out from the RetrievalDataset class
    """
    # img should be 3 channels
    desired_size = 100  # can be modified

    trans_resize = transforms.Resize([64, 64])
    trans_tograyscale = transforms.Grayscale(num_output_channels=1)
    trans_randomHorizontalFlip = transforms.RandomHorizontalFlip(
        p=0.5)  # 依概率p垂直翻转：transforms.RandomVerticalFlip(p=0.5)    #random flip horizontally or vertically, since original images are all vertical
    trans_randomVerticalFlip = transforms.RandomVerticalFlip(p=0.5)
    trans_totensor = transforms.ToTensor()
    trans_compose = transforms.Compose(
        [trans_resize, trans_tograyscale, trans_randomHorizontalFlip, trans_randomVerticalFlip, trans_totensor])
    img_out = trans_compose(img)

    trans_normalize = transforms.Normalize(
        mean=[img_out.mean()],
        std=[
            0.225])  # this is used for adjust contrast ratio, I think? But there is specific function to modify contrast ratio..

    # trans_padding    make sure it is 100*100

    return img_out  # output should be a 100*100 tensor

def CNN_output(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fypCNN = CNN(64)
    model_dict = torch.load(cfp + '/CNN/tensors.pt', map_location=torch.device('cpu'))
    fypCNN.load_state_dict(model_dict)
    fypCNN = fypCNN.to(device)

    img = Image.open(cfp + '/DrawRectangle/demo_img.png') # % (1))  # replace the image location

    trans_resize = transforms.Resize([64, 64])
    trans_tograyscale = transforms.Grayscale(num_output_channels=1)
    trans_totensor = transforms.ToTensor()
    trans_compose = transforms.Compose([trans_resize, trans_tograyscale, trans_totensor])

    img_out = trans_compose(img).to(device)
    img_mean = img_out.mean()
    std = 0.225
    img_out = (img_out - img_mean) / std
    img_out = img_out.unsqueeze(0)
    print((fypCNN(img_out)).detach().numpy())

    results = {}
    BoundingBoxSet = {}
    img_set = {}
    f0 = open(cfp + '/DrawRectangle/boundingBoxes/'+ 'demo_img.txt', 'r')
    line = f0.readline()
    i = 1
    while line != '':
        BoundingBoxSet[str(i)] = line.strip().split(',')[0:4] #dummy key as 0, maybe we can add another field
                                                 #at the end of each line of test0.txt to indicate
                                                 #parking bay id, then the key will be like:
                                                 #line.split(',')[4]
        boxCoordinates = BoundingBoxSet[str(i)]

        print(int(boxCoordinates[0]))

        img_set[str(i)] = img.crop(((int(boxCoordinates[0])), int(boxCoordinates[1]), int(boxCoordinates[2]), int(boxCoordinates[3])))
        i += 1
        line = f0.readline()
    for key in BoundingBoxSet:
        img = img_set[key]
        img_out = trans_compose(img).to(device)
        img_mean = img_out.mean()
        std = 0.225
        img_out = (img_out - img_mean) / std
        img_out = img_out.unsqueeze(0)
        img_set[key] = img_out

        results[key] = fypCNN(img_set[key])
        # endpoint = "https://127.0.0.1:8000/CNN_output"
        # requests.post(url=endpoint, data=results)
        f = open(cfp + '/CNN/CNN_result.txt', 'w')

        # for key in img_set:

    for result in results:
        f.write('Bounding Box ' + str(result) + ' with coordinates: ' + str(BoundingBoxSet[result])
                + ' produces result: ' + str(results[result].detach().numpy()[0, 0]) + '\n')
    f.close()
    return render(request,'base/draw_rectangles.html')
