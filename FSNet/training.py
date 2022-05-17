import os
import shutil
import torch
import numpy as np
from tqdm.autonotebook import tqdm

def train(model, train_dataloader, epochs, lr, epochs_till_chkpt,
          model_dir, loss_func, validation_dataloader=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=10, verbose=True)

    chkpts_dir = os.path.join(model_dir, 'checkpoints')
    if os.path.exists(chkpts_dir):
        val = input(f'The directory {chkpts_dir} already exists. Remove? [y/n]')
        if val == 'y':
            shutil.rmtree(chkpts_dir)
    
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = torch.tensor([])
        for epoch in range(epochs):
            epoch_loss = torch.tensor([])
            if not epoch % epochs_till_chkpt and epoch:
                torch.save(model.state_dict(),
                           os.path.join(chkpts_dir, f'model_epoch_{epoch}.pth'))
                np.savetxt(os.path.join(chkpts_dir, f'train_losses_epoch_{epoch}.txt'),
                           train_losses.cpu().numpy())

            for _, (model_input, gt) in enumerate(train_dataloader):
                model_input = model_input.float()
                model_input = model_input.cuda()
                gt = gt.cuda()

                model_output = model(model_input)
                loss = loss_func(model_output, gt)

                optim.zero_grad()
                loss.backward()
                optim.step()

                train_losses = torch.cat((train_losses,
                                          torch.tensor([loss])), 0)
                epoch_loss = torch.cat((epoch_loss,
                                        torch.tensor([loss])), 0)

                pbar.update(1)
                total_steps += 1

            if not epoch % epochs_till_chkpt:
                tqdm.write(f'Epoch {epoch}, loss: {np.mean(epoch_loss.cpu().numpy())}')

            if validation_dataloader != None:
                tqdm.write('Running validation set...')
                model.eval()
                with torch.no_grad():
                    val_losses = torch.tensor([])
                    for model_input, gt in validation_dataloader:
                        model_input = model_input.float()
                        model_input = model_input.cuda()
                        gt = gt.cuda()
                        model_output = model(model_input)
                        loss = loss_func(model_output, gt)
                        val_losses = torch.cat((val_losses,
                                                torch.tensor([loss])), 0)
                    scheduler.step(val_losses.mean())
                    tqdm.write(f'Validation loss after epoch {epoch}: {np.mean(val_losses.cpu().numpy())}')
                model.train()

        torch.save(model.state_dict(),
                   os.path.join(chkpts_dir, f'model_final.pth'))
        np.savetxt(os.path.join(chkpts_dir, f'train_losses_final.txt'),
                   np.array(train_losses))
