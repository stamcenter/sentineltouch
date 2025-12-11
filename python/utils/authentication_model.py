import torch.nn as nn
import torch

class AuthenticationCNN(nn.Module):
    def __init__(self,  num_classes: int = 10, fc1_width=7):
        super(AuthenticationCNN, self).__init__()

        self.num_classes = num_classes
        self.fc1_width = fc1_width
        conv_dropout_p = 0.4

        activation = nn.ReLU

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(32),
            activation(),
            nn.Dropout(conv_dropout_p),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(64),
            activation(),
            nn.Dropout(conv_dropout_p),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(128),
            activation(),
            nn.Dropout(conv_dropout_p),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(256),
            activation(),
            nn.Dropout(conv_dropout_p),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(512),
            activation(),
            nn.Dropout(conv_dropout_p),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 16),
            nn.BatchNorm1d(16),
            activation(),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def unit_norm(self, x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + epsilon)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        out1 = self.features(x[0])
        out2 = self.features(x[1])
        out1 = self.unit_norm(out1)
        out2 = self.unit_norm(out2)
        # print(f"x1 shape: {out1.shape}, x2 shape: {out2.shape}")
        # print(f"out1: {out1}")
        # print(f"out2: {out2}")
        out = self.classifier(out1 - out2)

        return out
    
    def embedding(self, x):
        x = self.features(x)
        x = self.unit_norm(x)
        return x
    
    def classify(self, x):
        x = self.classifier(x)
        return x