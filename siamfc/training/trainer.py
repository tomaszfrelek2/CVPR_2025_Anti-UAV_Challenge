import os
import torch
import time
import logging
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, criterion, optimizer, lr_scheduler, 
                 train_loader, val_loader, max_epochs, exp_name):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.exp_name = exp_name
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.writer = SummaryWriter(f"runs/{exp_name}")

        self.model_dir = f"models/{exp_name}"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self):
        logging.info(f"Starting training for {self.max_epochs} epochs")
        logging.info(f"Training on device: {self.device}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            train_loss = self._train_epoch(epoch)

            val_loss = self.validate()

            self.lr_scheduler.step()

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                }, f"{self.model_dir}/best_model.pth")

            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                }, f"{self.model_dir}/model_epoch_{epoch+1}.pth")
                
            logging.info(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(self.train_loader):
            template = batch['template'].to(self.device)
            search = batch['search'].to(self.device)
            label = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(template, search)

            loss = self.criterion(output, label)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}, Batch {i+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                template = batch['template'].to(self.device)
                search = batch['search'].to(self.device)
                label = batch['label'].to(self.device)

                output = self.model(template, search)

                loss = self.criterion(output, label)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss