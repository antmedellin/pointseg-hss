import os
import torch.utils.data
import torch
import torch.nn as nn
import logging
from datasets.libhsi import LIBHSIDataset
from datasets.pipelines import (
    Collect,
    GenerateHSICube,
    LoadENVIFromFile,
    BalancedHSICubes,
    LoadRGBFromFile,
)
from models import model_map
from functions import get_config, train, init_logger


def main():
    parser = get_config()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exp_lr", type=float, default=0.9)
    args = parser.parse_args()

    train_dataset = LIBHSIDataset(
        data_root=os.path.join("mmsegmentation",args.data_root, "train"),
        pipelines=[
            LoadENVIFromFile(),
            LoadRGBFromFile(),
            # BalancedHSICubes(
            #     num_classes=27, samples=args.samples, width=args.cube_width
            # ),
            GenerateHSICube(samples=args.samples, width=args.cube_width),
            Collect(keys=["cubes", "labels"]),
        ],
    )

    logger, work_dir = init_logger("train", args.model, args.cube_width, args.work_dir)
    ModelType = model_map[args.model]
    model = ModelType(204, 27, windows=args.cube_width)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=False,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.exp_lr)
    loss_fn = nn.CrossEntropyLoss(weight=train_dataset.class_weights, ignore_index=255)
    device = torch.device(args.device)
    model.to(device)
    loss_fn.to(device)
    for epoch in range(args.epochs):
        model, cost_time, loss = train(
            logger,
            epoch,
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            cube_bs=args.cube_bs,
        )
        torch.save(model.state_dict(), os.path.join(work_dir, f"epoch_{epoch}.pth"))
        logging.info(f"epoch:{epoch},cost_time:{cost_time // 60}m,loss:{loss}")

    logger.info("Training finished")


if __name__ == "__main__":
        
    main()
