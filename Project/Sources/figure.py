# Warning: Almost parts of this file is AI generated
from matplotlib import pyplot as plt

from data import parse_csv


def create_viz():
    cats = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

    samples = parse_csv("fashion-mnist_test.csv")

    picked = {}
    for img, cat in samples[:2000]:
        if cat not in picked:
            picked[cat] = img
        if len(picked) == 10:
            break

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            axes[i, j].imshow(picked[idx], cmap='gray')
            axes[i, j].set_title(cats[idx])
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('fashion_mnist_samples.png', dpi=150, bbox_inches='tight')
    plt.close()


def make_charts(loss_hist, acc, prec, rec, f1):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_hist, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    mets = ['Accuracy', 'Precision', 'Recall', 'F1']
    vals = [acc, prec, rec, f1]
    cols = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = plt.bar(mets, vals, color=cols)
    plt.title('Model Performance')
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)

    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(loss_hist, 'b-', linewidth=2, marker='o', markersize=4)
    plt.title('Training Loss Over Epochs', fontsize=14, weight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

