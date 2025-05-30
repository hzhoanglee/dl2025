from data import DataProc, parse_csv
from figure import make_charts, create_viz
from network import FashionNet


def main():
    dp = DataProc("hyper_param.txt")
    print(f"{len(dp.specs)} parameters")

    dataset = dp.load_data("fashion-mnist_train.csv", "fashion-mnist_test.csv")
    net = FashionNet(dataset['config'])
    loss_h = net.train_sess(dataset['training'])

    acc, prec, rec, f1 = net.eval_sess(dataset['testing'])
    print(f"Accuracy: {acc:.1f}%")
    print(f"Precision: {prec:.1f}%")
    print(f"Recall: {rec:.1f}%")
    print(f"F1: {f1:.1f}%")

    create_viz()
    make_charts(loss_h, acc, prec, rec, f1)


if __name__ == "__main__":
    main()
