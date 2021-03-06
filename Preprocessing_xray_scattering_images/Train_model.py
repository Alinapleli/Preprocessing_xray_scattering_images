from torch import nn
from torch.optim import Adam
from datetime import datetime as dt
from pathlib import Path


class PatchTrainer(object):
    def __init__(self, model, train_loader_1, val_loader_1, lr: float = 0.01):
        self.model = model
        self.train_loader = train_loader_1
        self.val_loader = val_loader_1
        self.optimizer = Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.losses = defaultdict(list)
        
    def train_epoch(self):
        self.model.train()
        
        losses = []
        for hist, values in zip(self.train_loader[0], self.train_loader[1]):
            if not hist.numel():
                continue

            self.optimizer.zero_grad()
            output = self.model(hist)

            loss = self.criterion(output, values)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            
        self.losses['Training loss'].append(np.mean(losses))
    
    @torch.no_grad()
    def test_val(self):
        self.model.eval()
        losses = []
        for hist, values in zip(self.val_loader[0], self.val_loader[1]):
            if not hist.numel():
                continue
            loss = self.criterion(self.model(hist), values)
            losses.append(loss.item())

        self.losses['Validation loss'].append(np.mean(losses))
        
    def plot_losses(self):
        clear_output(wait=True)
        for k, data in self.losses.items():
            plt.semilogy(data, label=k)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
    def train(self, epochs, plot_f: int = 100):
        state_dict = None
        best_model_epoch = 0
        min_val_loss = 1
        for i in range(epochs):
            self.train_epoch()
            self.test_val()
            if self.losses['Validation loss'][-1] < min_val_loss:
                min_val_loss = self.losses['Validation loss'][-1]
                best_model_epoch = i
                state_dict = self.model.cpu_state_dict()
            
            if i % plot_f == 0:
                self.plot_losses()
                plt.axvline(best_model_epoch, ls='--', color='red')
                plt.show()
        model.load_state_dict(state_dict)


def save_model(model, name: str):
    save_time = dt.now().strftime('%d %b %H:%M:%S')
    path = Path('models') / f'{name}_{save_time}.h5'
    torch.save(model.state_dict(), path)
    print('Saved model:', path)


def list_models():
    for path in Path('models').glob('*.h5'):
        print(path.name)


def load_model(model, name):
    model.load_state_dict(torch.load(Path('models') / name))
    return model
