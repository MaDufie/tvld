from Projects.AnomalyDetection.tvld.baselines_src.CANDI.models.build import build_model
from Projects.AnomalyDetection.tvld.baselines_src.CANDI.utils.parser import parse_args, load_config
from Projects.AnomalyDetection.tvld.baselines_src.CANDI.trainer import build_trainer
from Projects.AnomalyDetection.tvld.baselines_src.CANDI.utils.misc import mkdir, set_seeds, set_devices, set_wandb
from Projects.AnomalyDetection.tvld.baselines_src.CANDI.predictor import Predictor


def main():
    args = parse_args()
    cfg = load_config(args)

    # select cuda devices
    set_devices(cfg.VISIBLE_DEVICES)

    # setup wandb
    if cfg.WANDB.ENABLE:
        set_wandb(cfg)
    
    with open(mkdir(cfg.RESULT_DIR) / 'config.txt', 'w') as f:
        f.write(cfg.dump())

    # set random seed
    set_seeds(cfg.SEED)

    # build model
    model = build_model(cfg)

    # build trainer
    trainer = build_trainer(cfg, model)

    if cfg.TRAIN.ENABLE:
        trainer.train()
    if cfg.TEST.ENABLE:
        model = trainer.load_best_model()
        predictor = Predictor(cfg, model, tta=cfg.TEST.TTA.ENABLE)
        predictor.predict()
            
if __name__ == '__main__':
    main()
