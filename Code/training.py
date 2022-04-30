import os
import shutil
import torch
import numpy as np
import time
from tqdm.autonotebook import tqdm

def train(model, train_dataloader, epochs, lr, epochs_till_chkpt,
          steps_till_summary, model_dir, loss_func, validation_dataloader=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        resp = input(f'The model directory {model_dir} Exists! Remove? [y/n]')
        if resp == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    chkpts_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_till_chkpt and epoch:
                torch.save(model.state_dict(),
                           os.path.join(chkpts_dir, f'model_epoch_{epoch}.pth'))
                np.savetxt(os.path.join(chkpts_dir, f'train_losses_epoch_{epoch}.txt'),
                           np.array(train_losses))

            for _, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                model_input = model_input.cuda()
                gt = gt.cuda()

                model_output = model(model_input)
                loss = loss_func(model_output, gt)
                train_losses.append(loss)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if not total_steps % steps_till_summary:
                    tqdm.write(f'Epoch {epoch}, Total loss: {loss}, iteration time: {time.time() - start_time}')

                pbar.update(1)
                total_steps += 1

            if validation_dataloader != None:
                tqdm.write('Running validation set...')
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for model_input, gt in validation_dataloader:
                        model_output = model(model_input)
                        loss = loss_func(model_output, gt)
                        val_losses.append(loss)
                    tqdm.write(f'Validation loss after epoch {epoch}: {np.mean(val_losses)}')
                model.train()

        torch.save(model.state_dict(),
                   os.path.join(chkpts_dir, f'model_final.pth'))
        np.savetxt(os.path.join(chkpts_dir, f'train_losses_final.txt'),
                   np.array(train_losses))
