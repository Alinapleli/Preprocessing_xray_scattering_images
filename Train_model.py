from torch import nn
from torch.optim import Adam
from datetime import datetime as dt
from pathlib import Path


class PatchTrainer(object):
    def __init__(self, model, train_data, val_data, name: str, lr: float = 0.01):
        self.model = model
        self.name = name
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = Adam(self.model.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.val_criterion = nn.L1Loss(reduction='none')
        self.losses = defaultdict(list)

    def train_epoch(self):
        self.model.train()
        losses = []
        train_loader = self.train_data._get_batch(np.arange(len(self.train_data)))
        for hist, values in zip(train_loader[0], train_loader[1]):
            if not hist.numel():
                continue
            self.optimizer.zero_grad()
            output = self.model(hist)
            loss = self.criterion(output, values)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        self.losses['train'].append(np.mean(losses))

    @torch.no_grad()
    def test_val(self):
        self.model.eval()
        losses = []
        val_loader = self.val_data._get_batch(np.arange(len(self.val_data)))
        for hist, values in zip(val_loader[0], val_loader[1]):
            if not hist.numel():
                continue
            loss = self.criterion(self.model(hist), values)
            losses.append(loss.item())
        self.losses['val'].append(np.mean(losses))

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
            if self.losses['val'][-1] < min_val_loss:
                min_val_loss = self.losses['val'][-1]
                best_model_epoch = i
                state_dict = self.model.cpu_state_dict()
            if i % plot_f == 0:
                self.plot_losses()
                plt.axvline(best_model_epoch, ls='--', color='red')
                plt.show()
        model.load_state_dict(state_dict)
        save_model(model, self.name)


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
