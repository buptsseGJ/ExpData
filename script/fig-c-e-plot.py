
import pylab as pl


def drawROC(filename):
    evaluate_result = filename
    db = []  # [score,nonclk,clk]
    pos, neg = 0, 0
    with open(evaluate_result, "r") as fs:
        for line in fs:
            nonclk, clk, score = line.strip().split("\t")
            nonclk = int(nonclk)
            clk = int(clk)
            score = float(score)
            db.append([score, nonclk, clk])
            pos += clk
            neg += nonclk

    db = sorted(db, key=lambda x: x[0], reverse=True)

    # 计算ROC坐标点
    xy_arr = []
    tp, fp = 0.0, 0.0
    for i in range(len(db)):
        tp += db[i][2]
        fp += db[i][1]
        xy_arr.append([fp / neg, tp / pos])

    # 计算曲线下面积
    auc = 0.0
    prev_x = 0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x

    return xy_arr


def diff(p, q):
    from scipy import interpolate

    pfn = interpolate.interp1d(p[0], p[1])
    qfn = interpolate.interp1d(q[0], q[1])
    xs = set(p[0])
    xs.update(q[0])
    xs = sorted(xs)[1:-1]
    ys = [pfn(x) - qfn(x) for x in xs]
    return xs, ys


def draw_compared(ax, path, name):
    xs, ys = zip(*drawROC(f"{path}-tii"))
    ax.plot(xs, ys, label=f"{name}:TII", linestyle="--")
    xs, ys = zip(*drawROC(f"{path}-tse"))
    ax.plot(xs, ys, label=f"{name}:TSE", linestyle=":")


def draw_diff(ax, path, name):
    p = zip(*drawROC(f"{path}-tii"))
    q = zip(*drawROC(f"{path}-tse"))
    xs, ys = diff(tuple(p), tuple(q))
    ax.plot(xs, ys, label=f"{name}")


def draw_topic(drawfn, ax, xs, pfx, var):
    for i in xs:
        drawfn(ax, f"../{pfx}-{i}", f"{var}={i}")
    drawfn(ax, f"../G", "Gemini")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="best")


def draw_all(drawfn):
    fig, axes = pl.subplots(nrows=1, ncols=3)

    draw_topic(drawfn, axes[0], [16, 64, 128, 256], "size", "p")
    draw_topic(drawfn, axes[1], [1, 2, 3, 4, 5], "depth", "n")
    draw_topic(drawfn, axes[2], [1, 2, 4, 6, 8], "iteration", "T")
    axes[0].set_title("(c) ROC curves for different embedding size p")
    axes[1].set_title("(d) ROC curves for the embedding depth n")
    axes[2].set_title("(e) ROC curves for the number of iterations T")

    fig.set_size_inches(15, 4)
    fig.tight_layout()
    return fig, axes


def main():
    fig, axes = draw_all(draw_compared)
    fig.savefig("combine.pdf")

    fig, axes= draw_all(draw_diff)
    for ax in axes:
        ax.set_ylabel("Difference")
    fig.savefig("diff.pdf")


if __name__ == "__main__":
    main()
