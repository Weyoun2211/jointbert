import torch

# def accuracy(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def accuracy(output, target):
    with torch.no_grad():
        pred = (output>0.5).long()
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def tp(output, target):
    with torch.no_grad():
        pred = (output>0.5).long()
        assert pred.shape[0] == len(target)
        tp = 0
        tp = torch.sum((pred == 1) & (target == 1)).item()
        return tp

def fp(output, target):
    with torch.no_grad():
        pred = (output>0.5).long()
        assert pred.shape[0] == len(target)
        fp = 0
        fp = torch.sum((pred == 1) & (target == 0)).item()
        return fp

def tn(output, target):
    with torch.no_grad():
        pred = (output>0.5).long()
        assert pred.shape[0] == len(target)
        tn = 0
        tn = torch.sum((pred == 0) & (target == 0)).item()
        return tn

def fn(output, target):
    with torch.no_grad():
        pred = (output>0.5).long()
        assert pred.shape[0] == len(target)
        fn = 0
        fn = torch.sum((pred == 0) & (target == 1)).item()
        return fn

def accuracy_multi1(output, target):
    with torch.no_grad():
        pred = output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_multi2(output, target):
    with torch.no_grad():
        pred = output
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def tp_multibin(output1, target1, output2, target2):
    with torch.no_grad():
        assert output1.shape[0] == len(target1)
        assert output2.shape[0] == len(target2)
        tp = 0
        tp = torch.sum((output1 == output2) & (target1 == target2)).item()
        return tp

def fp_multibin(output1, target1, output2, target2):
    with torch.no_grad():
        assert output1.shape[0] == len(target1)
        assert output2.shape[0] == len(target2)
        fp = 0
        fp = torch.sum((output1 == output2) & (target1 != target2)).item()
        return fp

def tn_multibin(output1, target1, output2, target2):
    with torch.no_grad():
        assert output1.shape[0] == len(target1)
        assert output2.shape[0] == len(target2)
        tn = 0
        tn = torch.sum((output1 != output2) & (target1 != target2)).item()
        return tn

def fn_multibin(output1, target1, output2, target2):
    with torch.no_grad():
        assert output1.shape[0] == len(target1)
        assert output2.shape[0] == len(target2)
        fn = 0
        fn = torch.sum((output1 != output2) & (target1 == target2)).item()
        return fn