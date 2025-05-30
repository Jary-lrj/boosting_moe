from trainer import train_loop
from dataset import get_dataloaders
from model import SASRec
from args import Args

args = Args()
train_loader, valid_loader, test_loader = get_dataloaders(args)
model = SASRec(args.user_num, args.item_num, args).to(args.device)
train_loop(model, train_loader, valid_loader, test_loader, args)
