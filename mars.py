import sys
from engine.engine import MarsEngine


if __name__ == "__main__":
    mode = "pipe"  
    #mode = "eval"
    nobuf = True

    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg == "-nobuf":
            nobuf = True
        elif arg == "-train":
            mode = "train"
        elif arg == "-eval":
            mode = "eval"
        elif arg == "-pipe":
            mode = "pipe"

    MarsEngine(
        mode=mode,
        #cfgname="c1.nano.full",
        #cfgname="c1.nano.full_pretrain",
        #cfgname="c1.nano.teacher",
        #cfgname="c1.nano.distillation",
        #cfgname="c1.nano.swin_transformer",
        #cfgname="c1.nano.mosaic",
        root="/home/zxh/mars/run_root", # 注意项目运行root不要放在代码路径下
        nobuf=nobuf,
    ).run()
