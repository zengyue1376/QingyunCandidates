import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MmEmbDataset
from model_rqvae import RQVAE


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--device', default='cuda')

    parser.add_argument('--mm_emb_id', nargs='+', default=['83'], type=str)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')

    return parser.parse_args()


def train():
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    writer = SummaryWriter(args.log_dir)

    dataset = MmEmbDataset(os.environ.get("TRAIN_DATA_PATH"), args.mm_emb_id)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = RQVAE(
        input_dim=3584,
        hidden_channels=[512, 256],
        latent_dim=64,
        num_codebooks=2,
        codebook_size=[64, 64],
        shared_codebook=False,
        kmeans_method='bkmeans',
        kmeans_iters=20,
        distances_method='l2',
        loss_beta=0.25,
        device=args.device,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Start Training...")
    global_step = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss_sum = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = batch.to(args.device)
            x_hat, semantic_ids, recon_loss, rqvae_loss, total_loss = model(x)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()
            writer.add_scalar('Loss/train', total_loss.item(), global_step)
            global_step += 1

        avg_train_loss = train_loss_sum / len(train_loader)
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in valid_loader:
                x = batch.to(args.device)
                x_hat, semantic_ids, recon_loss, rqvae_loss, total_loss = model(x)
                val_loss_sum += total_loss.item()

        avg_val_loss = val_loss_sum / len(valid_loader)
        writer.add_scalar('Loss/valid', avg_val_loss, epoch)
        print(f"[Epoch {epoch}] Avg Valid Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        ckpt_path = Path(args.ckpt_dir) / f"epoch={epoch}_loss={avg_val_loss:.4f}.pt"
        torch.save(model.state_dict(), ckpt_path)

    writer.close()
    print("Training finished.")


if __name__ == '__main__':
    train()
