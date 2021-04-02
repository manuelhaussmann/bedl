import fire
import torch as th
from cifar5 import create_cifar5
from exp_classification_archs import *
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import Phi


def prepare_data(dataset="mnist"):
    kwargs = {'num_workers': 1, 'pin_memory': True} if iscuda else {}

    if dataset == "mnist":
        n_channels, n_classes = 1, 10
        train_loader = DataLoader(datasets.MNIST('../data/mnist/', train=True, download=True,
                                                 transform=transforms.Compose([transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,), (0.3081,))])),
                                  batch_size=100, shuffle=True, **kwargs)

        test_loader = DataLoader(datasets.MNIST('../data/mnist/', train=False,
                                                transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                 batch_size=100, shuffle=False, **kwargs)

        ood_loader = DataLoader(datasets.FashionMNIST('../data/fashion/', train=True, download=True,
                                                      transform=transforms.Compose([transforms.ToTensor(),
                                                                                      transforms.Normalize((0.28604,),(0.35302,))])),
                                batch_size=100, shuffle=False, **kwargs)

    elif dataset == "cifar5":
        n_channels, n_classes = 3, 5
        train_data, test_data, ood_data = create_cifar5()
        train_loader = DataLoader(train_data, batch_size=100, shuffle=True, **kwargs)
        test_loader = DataLoader(test_data, batch_size=100, shuffle=False, **kwargs)
        ood_loader = DataLoader(ood_data, batch_size=100, shuffle=False, **kwargs)

    return train_loader, test_loader, ood_loader, n_channels, n_classes


# The training
def train(model, loader, optimizer, epoch, verbose=False):
    model.train()
    losses = 0
    for data, target in tqdm(loader, leave=False, desc=f"Train Epoch {epoch}"):
        if iscuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        loss = model.loss(data, target, len(loader.dataset))
        loss.backward()
        losses += loss.item()
        optimizer.step()
    if verbose:
        tqdm.write(f"{epoch}: AvgLoss = {losses / len(loader):.02f}")


def test(model, loader, epoch, n_classes, label="Test"):
    model.eval()
    correct = 0
    with th.no_grad():
        for data, target in tqdm(loader, leave=False, desc=f"Test Epoch {epoch}"):
            if iscuda:
                data, target = data.cuda(), target.cuda()
            probs = model.predict(data)
            pred = probs.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
        # tqdm.write(f"{probs.max(1)[0].data}")
    tqdm.write(f"{epoch}: {label} set: Accuracy: {correct}/{len(loader.dataset)} ({100. * correct / len(loader.dataset):.0f}%) Error(%) = {100*( 1 - correct/len(loader.dataset)):.02f}")


def entropy_all(p):
    return - (p * th.log(p + 1e-8)).sum(1)

def get_probs(net, loader):
    probs = []
    with th.no_grad():
        for data, target in loader:
            if iscuda:
                data = data.cuda()
            probs.append(net.predict(data))
        return th.cat(probs)

def main(save_name="", train_bedl=False, train_bedlpac=False, train_drop=False, train_vb=False, train_dvi=False, dataset="mnist"):
    # Get dataset
    train_loader, test_loader, ood_loader, n_channels, n_classes = prepare_data(dataset)

    if dataset == "mnist":
        max_epochs = 1
        lrate = 1e-3 # Default for Adam
    elif dataset == "cifar5":
        max_epochs = 40
        lrate = 1e-3

    if train_bedl:
        tqdm.write("## bedl")
        model = BEDL(n_channels, n_classes).cuda() if iscuda else BEDL(n_channels, n_classes)
        print(model)
        optimizer = th.optim.Adam(model.parameters(), lr=lrate)

        # The training
        for epoch in tqdm(range(max_epochs), leave=False):
            train(model, train_loader, optimizer, epoch)
            # test(model, test_loader, epoch, n_classes, " Test")
        test(model, train_loader, epoch, n_classes, " Train")
        test(model, test_loader, epoch, n_classes, " Test")
        test(model, ood_loader, epoch, n_classes, " OOD")
        bedl_entr = entropy_all(get_probs(model, ood_loader)).cpu()
        bedl_entr_tr = entropy_all(get_probs(model, train_loader)).cpu()
        bedl_entr_te = entropy_all(get_probs(model, test_loader)).cpu()

    if train_bedlpac:
        tqdm.write("## bedlPAC")
        model = BEDLPAC(n_channels, n_classes).cuda() if iscuda else BEDLPAC(n_channels, n_classes)
        print(model)
        optimizer = th.optim.Adam(model.parameters(), lr=lrate)

        # The training
        for epoch in tqdm(range(max_epochs), leave=False):
            train(model, train_loader, optimizer, epoch, verbose=False)
            # test(model, test_loader, epoch, n_classes, " Test")
        test(model, train_loader, epoch, n_classes, " Train")
        test(model, test_loader, epoch, n_classes, " Test")
        test(model, ood_loader, epoch, n_classes, " OOD")
        bedlpac_entr = entropy_all(get_probs(model, ood_loader)).cpu()
        bedlpac_entr_tr = entropy_all(get_probs(model, train_loader)).cpu()
        bedlpac_entr_te = entropy_all(get_probs(model, test_loader)).cpu()

    if train_drop:
        tqdm.write("## Dropout")
        model = DropNet(n_channels, n_classes).cuda() if iscuda else DropNet(n_channels, n_classes)
        optimizer = th.optim.Adam(model.parameters(), lr=lrate)

        # The training
        for epoch in tqdm(range(max_epochs), leave=False):
            train(model, train_loader, optimizer, epoch)
        test(model, train_loader, epoch, n_classes, " Train")
        test(model, test_loader, epoch, n_classes, " Test")
        test(model, ood_loader, epoch, n_classes, " OOD")
        drop_entr = entropy_all(get_probs(model, ood_loader)).cpu()
        drop_entr_tr = entropy_all(get_probs(model, train_loader)).cpu()
        drop_entr_te = entropy_all(get_probs(model, test_loader)).cpu()

    if train_vb:
        tqdm.write("## VarOut")
        model = VBNet(n_channels, n_classes).cuda() if iscuda else VBNet(n_channels, n_classes)
        optimizer = th.optim.Adam(model.parameters(), lr=lrate)

        # The training
        for epoch in tqdm(range(max_epochs), leave=False):
            train(model, train_loader, optimizer, epoch)
        test(model, train_loader, epoch, n_classes, " Train")
        test(model, test_loader, epoch, n_classes, " Test")
        test(model, ood_loader, epoch, n_classes, " OOD")
        vb_entr = entropy_all(get_probs(model, ood_loader)).cpu()
        vb_entr_tr = entropy_all(get_probs(model, train_loader)).cpu()
        vb_entr_te = entropy_all(get_probs(model, test_loader)).cpu()

    if train_dvi:
        tqdm.write("## DVI")
        model = DVINet(n_channels, n_classes).cuda() if iscuda else DVINet(n_channels, n_classes)
        optimizer = th.optim.Adam(model.parameters(), lr=lrate)

        # The training
        for epoch in tqdm(range(max_epochs), leave=False):
            train(model, train_loader, optimizer, epoch)

        test(model, train_loader, epoch, n_classes, " Train")
        test(model, test_loader, epoch, n_classes, " Test")
        test(model, ood_loader, epoch, n_classes, " OOD")
        dvi_entr = entropy_all(get_probs(model, ood_loader)).cpu()
        dvi_entr_tr = entropy_all(get_probs(model, train_loader)).cpu()
        dvi_entr_te = entropy_all(get_probs(model, test_loader)).cpu()



    if True:
        from statsmodels.distributions import ECDF
        from sklearn.metrics import auc
        import seaborn as sns
        import numpy as np
        import matplotlib.pyplot as plt
        sns.set_style("whitegrid")

        def calc_ent_auc(ent):
            # via https://github.com/pawni/BayesByHypernet/
            max_ent = np.log(n_classes)
            hist, bin_edges = np.histogram(ent, density=True, bins=np.arange(0, max_ent, max_ent / 500))
            
            c_hist = np.cumsum(hist * np.diff(bin_edges))

            return np.sum(np.diff(bin_edges) * c_hist)  

        if train_bedl:
            bedl_ecdf = ECDF(bedl_entr)
            bedl_ecdf.x[0] = 0
            print(f"bedl: {auc(bedl_ecdf.x,bedl_ecdf.y):.2f}")
            print(f"bedl: {calc_ent_auc(bedl_entr.numpy()):.2f}//{calc_ent_auc(bedl_entr_te.numpy()):.2f}//{calc_ent_auc(bedl_entr_tr.numpy()):.2f}")
            print(f"bedl: {calc_ent_auc(bedl_entr.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(bedl_entr_te.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(bedl_entr_tr.numpy())/math.log(n_classes):.2f}")
            # plt.plot(bedl_ecdf.x, bedl_ecdf.y, label="bedl")

            sns.distplot(bedl_entr_tr.view(-1).numpy())
            sns.distplot(bedl_entr_te.view(-1).numpy())
            sns.distplot(bedl_entr.view(-1).numpy())
            plt.show()



        if train_bedlpac:
            bedlpac_ecdf = ECDF(bedlpac_entr)
            bedlpac_ecdf.x[0] = 0
            print(f"bedlPAC: {auc(bedlpac_ecdf.x,bedlpac_ecdf.y):.2f}")
            print(f"bedlPAC: {calc_ent_auc(bedlpac_entr.numpy()):.2f}")
            print(f"bedlPAC: {calc_ent_auc(bedlpac_entr.numpy())/math.log(n_classes):.2f}")
            # plt.plot(bedlpac_ecdf.x, bedlpac_ecdf.y, label="bedlpac")
            print(f"bedlPAC: {calc_ent_auc(bedlpac_entr.numpy()):.2f}//{calc_ent_auc(bedlpac_entr_te.numpy()):.2f}//{calc_ent_auc(bedlpac_entr_tr.numpy()):.2f}")
            print(f"bedlPAC: {calc_ent_auc(bedlpac_entr.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(bedlpac_entr_te.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(bedlpac_entr_tr.numpy())/math.log(n_classes):.2f}")

            # sns.distplot(bedlpac_entr_tr.view(-1).numpy())
            # sns.distplot(bedlpac_entr_te.view(-1).numpy())
            # sns.distplot(bedlpac_entr.view(-1).numpy())
            # plt.savefig("results/run-latent1.png")
            # plt.show()

        if train_drop:
            drop_ecdf = ECDF(drop_entr)
            drop_ecdf.x[0] = 0
            print(f"DROP: {auc(drop_ecdf.x,drop_ecdf.y):.2f}")
            print(f"DROP: {calc_ent_auc(drop_entr.numpy()):.2f}")
            print(f"DROP: {calc_ent_auc(drop_entr.numpy())/math.log(n_classes):.2f}")
            # plt.plot(drop_ecdf.x, drop_ecdf.y, label="drop")
            print(f"DROP: {calc_ent_auc(drop_entr.numpy()):.2f}//{calc_ent_auc(drop_entr_te.numpy()):.2f}//{calc_ent_auc(drop_entr_tr.numpy()):.2f}")
            print(f"DROP: {calc_ent_auc(drop_entr.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(drop_entr_te.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(drop_entr_tr.numpy())/math.log(n_classes):.2f}")

        if train_vb:
            vb_ecdf = ECDF(vb_entr)
            vb_ecdf.x[0] = 0
            print(f"VB:  {auc(vb_ecdf.x,vb_ecdf.y):.2f}")
            print(f"VB: {calc_ent_auc(vb_entr.numpy()):.2f}")
            print(f"VB: {calc_ent_auc(vb_entr.numpy())/math.log(n_classes):.2f}")
            # plt.plot(vb_ecdf.x, vb_ecdf.y, label="vb")
            print(f"VB: {calc_ent_auc(vb_entr.numpy()):.2f}//{calc_ent_auc(vb_entr_te.numpy()):.2f}//{calc_ent_auc(vb_entr_tr.numpy()):.2f}")
            print(f"VB: {calc_ent_auc(vb_entr.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(vb_entr_te.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(vb_entr_tr.numpy())/math.log(n_classes):.2f}")

        if train_dvi:
            dvi_ecdf = ECDF(dvi_entr)
            dvi_ecdf.x[0] = 0
            print(f"DVI: {auc(dvi_ecdf.x,dvi_ecdf.y):.2f}")
            print(f"DVI: {calc_ent_auc(dvi_entr.numpy()):.2f}")
            print(f"DVI: {calc_ent_auc(dvi_entr.numpy())/math.log(n_classes):.2f}")
            # plt.plot(dvi_ecdf.x, dvi_ecdf.y, label="dvi")
            print(f"DVI: {calc_ent_auc(dvi_entr.numpy()):.2f}//{calc_ent_auc(dvi_entr_te.numpy()):.2f}//{calc_ent_auc(dvi_entr_tr.numpy()):.2f}")
            print(f"DVI: {calc_ent_auc(dvi_entr.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(dvi_entr_te.numpy())/math.log(n_classes):.2f}//{calc_ent_auc(dvi_entr_tr.numpy())/math.log(n_classes):.2f}")




if __name__ == "__main__":
    if th.cuda.is_available():
        th.backends.cudnn.benchmark = True
        iscuda = True
    else:
        iscuda = False

    fire.Fire(main)
