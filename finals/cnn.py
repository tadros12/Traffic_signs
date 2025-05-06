class GTSRBDataset(Dataset):
    def __init__(self, root_dir, load_type='Train', resize_size=(64, 64), transform=None, use_features=True):
        self.root_dir = root_dir
        self.load_type = load_type
        self.resize_size = resize_size
        self.transform = transform
        self.use_features = use_features
        self.data = []
        self._load_data()

    def _load_data(self):
        if self.load_type == "Train":
            class_dirs = sorted(os.listdir(self.root_dir))
            for class_id in class_dirs:
                class_dir = os.path.join(self.root_dir, class_id)
                class_label = int(class_id)
                annotation_file = os.path.join(class_dir, f'GT-{class_id}.csv')
                annotations = pd.read_csv(annotation_file, sep=';')

                for _, row in annotations.iterrows():
                    self.data.append({
                        'img_path': os.path.join(class_dir, row['Filename']),
                        'label': class_label,
                        'bbox': {
                            'x1': row['Roi.X1'], 'y1': row['Roi.Y1'],
                            'x2': row['Roi.X2'], 'y2': row['Roi.Y2'],
                            'original_width': row['Width'], 'original_height': row['Height']
                        }
                    })
        elif self.load_type == "Test":
            annotation_file = "/home/theodoros/dataset/GT-final_test.csv"
            annotations = pd.read_csv(annotation_file, sep=';')

            for _, row in annotations.iterrows():
                self.data.append({
                    'img_path': os.path.join(self.root_dir, row['Filename']),
                    'label': int(row['ClassId']),
                    'bbox': {
                        'x1': row['Roi.X1'], 'y1': row['Roi.Y1'],
                        'x2': row['Roi.X2'], 'y2': row['Roi.Y2'],
                        'original_width': row['Width'], 'original_height': row['Height']
                    }
                })

    def _process_image(self, img_path):
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, self.resize_size)
        img_norm = img_resize / 255.0
        gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

        if self.use_features:
            hog_desc, hog_image = hog(
                gray,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualize=True,
                feature_vector=True
            )

            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            return img_norm, keypoints, descriptors, hog_desc, hog_image
        else:
            return img_norm, None, None, None, None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['img_path']
        label = item['label']
        bbox = item['bbox']

        img_norm, keypoints, descriptors, hog_desc, hog_image = self._process_image(img_path)

        image_tensor = torch.tensor(img_norm, dtype=torch.float).permute(2, 0, 1)  ## CHW format

        if self.transform:
            image_tensor = self.transform(image_tensor)

        sample = {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'bbox': bbox
        }

        if self.use_features:
            if descriptors is None:
                descriptors = np.zeros((1, 128), dtype=np.float32)
            sample.update({
                'sift_features': descriptors,
                'hog_features': torch.tensor(hog_desc, dtype=torch.float),
                'hog_image': torch.tensor(hog_image, dtype=torch.float)
            })

        return sample
class SimpleResCNN(nn.Module):
    def __init__(self, num_classes=43):  ## Our dataset (GTSRB) has 43 classes
        super(SimpleResCNN, self).__init__()
        
        ## (first convvv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  ## 64 -> 32
        )
        
        ## (second convvv + skip conn)
        self.conv2_main = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2_skip = nn.Conv2d(32, 64, kernel_size=1)

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  ## 32 -> 16

        ## (another final conv, like a combiner one)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  ## 16 -> 8
        )

        ## Classifier (final step)
        self.gap = nn.AdaptiveAvgPool2d(1)  ## 8x8 -> 1x1
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        ## step 1
        x = self.conv1(x)  ## (3, 64, 64) -> (32, 32, 32)

        ## step 2
        identity = self.conv2_skip(x)  ## (32, 32, 32) -> (64, 32, 32)
        out = self.conv2_main(x)
        x = self.relu2(out + identity)  ## Residual conn
        x = self.pool2(x)  ## -> (64, 16, 16)

        ## step 3
        x = self.conv3(x)  ## -> (128, 8, 8)
        
        ## step 4
        x = self.gap(x).view(x.size(0), -1)  ## -> (128,)
        out = self.fc(x)
        
        return out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = GTSRBDataset(root_dir=train_path, load_type='Train', use_features=False)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_dataset = GTSRBDataset(root_dir=test_path, load_type='Test', use_features=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4)
model = SimpleResCNN(num_classes=43).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 5

for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100. * correct_train / total_train)

        # Validation block
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.inference_mode():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = 100. * correct_val / total_val
        print(f"\nEpoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {100. * correct_train / total_train:.2f}%, "
              f"Val Acc: {val_acc:.2f}%, "
              f"Val Loss: {val_loss / len(test_loader):.4f}")