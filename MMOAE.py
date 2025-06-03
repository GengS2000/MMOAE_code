# Multi-omics Deep Representation Learning Example
# Task 1: Pretraining and Visualization

import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loading and preprocessing
def load_multiomics_data(filepath):
    data = pd.read_csv(filepath)
    CA = data['CA_type']
    labels = np.array(CA)
    data = data.drop(columns=['CA_type'])
    return data, labels

def prepare_tensor(data):
    X = data.to_numpy(dtype=np.float32)
    return torch.tensor(X)

# Autoencoder architecture
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2000), nn.BatchNorm1d(2000), nn.LeakyReLU(),
            nn.Linear(2000, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU(),
            nn.Linear(1000, 256), nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.LeakyReLU(),
            nn.Linear(256, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU(),
            nn.Linear(1000, 2000), nn.BatchNorm1d(2000), nn.LeakyReLU(),
            nn.Linear(2000, input_dim)
        )

    def forward(self, x):
        return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = Encoder(input_dim)
        self.decoder = Decoder(input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Training utility
class TensorDatasetWrapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
        

def masked_mse_loss(output, target,mask):
    # Compute loss only on masked regions (where noise is introduced)
    loss = (output - target) ** 2
    loss = loss * (mask)  # Apply mask to focus on noisy regions
    return loss.sum() / (mask).sum()  # Normalize by the number of masked (noisy) values
    
def train_autoencoder(model, train_loader, val_loader, nepochs=500, patience=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    best_loss = float('inf')
    no_improve = 0

    for epoch in range(nepochs):
        model.train()
        train_loss = 0
        for i in train_loader:
            fea,lab = i
            fea = fea.to(device)
            lab = lab.to(device)
            zeromask = (lab != 0).float().to(device)
            #lab = lab.squeeze(1)  此处lab为X_initial
            outputs = model(fea)
            re_loss = masked_mse_loss(outputs, lab, mask=zeromask)
            if torch.cuda.is_available():
                re_loss = re_loss.to(device)
            goptimizer.zero_grad()
            re_loss.backward()
            goptimizer.step()
            train_loss = re_loss+train_loss

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for fea, lab in val_loader:
                fea,lab = i
                fea = fea.to(device)
                lab = lab.to(device)
                zeromask = (lab != 0).float().to(device)
                #lab = lab.squeeze(1)
                outputs= model(fea)
                val_loss += masked_mse_loss(outputs, lab,zeromask).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_autoencoder.pth')
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping.")
            break

# Dimensionality reduction and clustering

def cluster_and_plot(X_encoded, labels, method_name="MMOAE"):
    tsne = TSNE(n_components=2, perplexity=70)
    X_tsne = tsne.fit_transform(X_encoded)
    kmeans = KMeans(n_clusters=len(np.unique(labels)), random_state=42)
    cluster_labels = kmeans.fit_predict(X_tsne)

    plt.figure(figsize=(8, 6))
    plt.title(f"{method_name} t-SNE Clustering")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab20', s=10, alpha=0.6)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(f"{method_name}_tsne_plot.png", dpi=300)
    plt.show()

    scores = {
        'Silhouette': silhouette_score(X_tsne, cluster_labels),
        'DBI': davies_bouldin_score(X_tsne, cluster_labels),
        'CH': calinski_harabasz_score(X_tsne, cluster_labels),
        'ARI': adjusted_rand_score(labels, cluster_labels),
        'Homogeneity': homogeneity_score(labels, cluster_labels),
        'Completeness': completeness_score(labels, cluster_labels),
        'V-Measure': v_measure_score(labels, cluster_labels),
    }

    for k, v in scores.items():
        print(f"{k}: {v:.4f}")

# Main pipeline
if __name__ == '__main__':
    data, CA_label = load_multiomics_data('multiomics.csv')
    X_tensor = prepare_tensor(data)
    dataset = TensorDatasetWrapper(X_tensor, X_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = Autoencoder(input_dim=X_tensor.shape[1]).to(device)
    train_autoencoder(model, train_loader, val_loader)

    model.load_state_dict(torch.load('best_autoencoder.pth'))
    model.eval()
    with torch.no_grad():
        X_encoded = model.encoder(X_tensor.to(device)).cpu().numpy()

    cluster_and_plot(X_encoded, CA_label, method_name="MMOAE")



# === Task 2: Fine-tuning for Cancer Subtype Classification (BRCA, METABRIC) ===

class SubtypeClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits

def save_finetuned_model(classifier, path):
    torch.save(classifier.state_dict(), path)

def evaluate_classifier(classifier, X_tensor, y_tensor, label_decoder=None):
    classifier.eval()
    with torch.no_grad():
        _, logits = classifier(X_tensor)
        pred = torch.argmax(logits, dim=1)
        from sklearn.metrics import classification_report
        y_true = y_tensor.cpu().numpy()
        y_pred = pred.cpu().numpy()
        if label_decoder is not None:
            y_true = label_decoder.inverse_transform(y_true)
            y_pred = label_decoder.inverse_transform(y_pred)
        print(classification_report(y_true, y_pred, digits=4))

# TCGA BRCA fine-tuning example

def finetune_brca(model_path='best_autoencoder.pth', label_file='tcgaphenotype/tcgabrcaphonetype.csv'):
    data = pd.read_csv('multiomics.csv')
    label = pd.read_csv(label_file)
    label = label[['sample', 'PAM50Call_RNAseq']].dropna()
    X = data[data['CA_type'] == 'BRCA']
    X = pd.merge(X, label, on='sample').dropna()
    y = X['PAM50Call_RNAseq']
    X = X.drop(columns=['sample', 'CA_type', 'PAM50Call_RNAseq'])
    X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(device)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    encoder = Autoencoder(X_tensor.shape[1]).to(device)
    encoder.load_state_dict(torch.load(model_path))
    classifier = SubtypeClassifier(encoder.encoder).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-4)

    for epoch in range(30):
        classifier.train()
        for fea, lab in train_loader:
            features, logits = classifier(fea)
            loss = criterion(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        acc = 0
        total = 0
        with torch.no_grad():
            for fea, lab in val_loader:
                _, logits = classifier(fea)
                pred = torch.argmax(logits, dim=1)
                acc += (pred == lab).sum().item()
                total += len(lab)
        print(f"Epoch {epoch+1} validation accuracy: {acc / total:.4f}")

    save_finetuned_model(classifier, 'finetuned_brca_classifier.pth')
    return classifier
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(23904, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(2000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(1000, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(256, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(1000, 2000)
        self.bn3 = nn.BatchNorm1d(2000)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(2000, 23904)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

MMOAE = Autoencoder()
pretrain_encoder = MMOAE.encoder
class SubtypeClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super(SubtypeClassifier, self).__init__()
        self.encoder = encoder  
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)  
        logits = self.classifier(features)
        return features, logits
        
subtypeclass_model = SubtypeClassifier(encoder=pretrain_encoder).to(device)

subtypeclass_model.load_state_dict(torch.load('finetuned_brca_classifier.pth', weights_only=True))
BRCA_scrach_encoded_feature = []
nonpretrain_encoded_labels = []
with torch.no_grad():
    fea = subtypeclass_model.encoder(X_tensor)
BRCA_scrach_encoded_feature.append(fea)

BRCA_scrach_encoded_features = torch.cat(BRCA_scrach_encoded_feature, dim=0)
BRCA_scrach_X_encoded = BRCA_scrach_encoded_features.cpu().numpy()
BRCA_df = pd.DataFrame(BRCA_scrach_X_encoded)
BRCA_df['subtype'] = subtype

X = BRCA_df.drop('subtype', axis=1)
y = BRCA_df['subtype']


X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,  
    random_state=42
)


svm_model = svm.SVC(kernel='rbf', decision_function_shape='ovr')  
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_val)

print(classification_report(y_val, y_pred, digits=4))
report_dict = classification_report(y_val, y_pred, digits=4, output_dict=True)


df_report = pd.DataFrame(report_dict).transpose().reset_index()
df_report.columns = ['Class'] + list(df_report.columns[1:]) 


subtype_labels = ['Basal', 'Her2', 'LumA', 'LumB', 'Normal']  
if len(subtype_labels) == 5:
    df_report['Class'] = df_report['Class'].astype(str)
    df_report.loc[:4, 'Class'] = subtype_labels  


df_report.round(4).to_csv('BRCA_molecular_subtype_finetune/brca_finetune.csv', index=False)

# === Task 3: Survival Prediction (e.g., COADREAD) ===

class SurvivalPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

def cox_loss(pred_risk, durations, events):
    hazards = pred_risk.squeeze()
    idx = torch.argsort(durations, descending=True)
    hazards = hazards[idx]
    events = events[idx]
    log_cum_hazard = torch.logcumsumexp(hazards, dim=0)
    loss = -(hazards - log_cum_hazard) * events
    return loss.mean()

# COADREAD survival fine-tuning example
feature = pd.read_csv('multiomics.csv')
print(feature.shape)
survival = pd.read_csv('survival.csv')
print(survival.shape)

OS = survival.drop(columns=['DSS.time','PFI.time','DFI.time','DSS','PFI','DFI'])
#feature = feature.drop(feature.columns[0], axis=1)
filtered_COADREAD = feature[feature['CA_type'] == 'COADREAD']
X = filtered_COADREAD.drop(columns=['CA_type'])

merged_data = pd.merge(X, OS, on='sample', how='inner')

print(merged_data.shape)
merged_data = merged_data.dropna()


OStime = merged_data['OS.time']
OSevent = merged_data['OS']

X_df = merged_data.drop(columns=['OS.time','sample','OS'])
X_df = X_df.to_numpy()
X_tensor = torch.tensor(X_df, dtype=torch.float32)

val_data = pd.read_csv('0413/maskcoadread.csv')
print(val_data.head())
print(val_data.shape)
val_data = pd.merge(val_data, OS, on='sample', how='inner')
val_data = val_data.dropna()
val_OStime = val_data['OS.time']
val_event = val_data['OS']
valdf = val_data.drop(columns=['OS.time','sample','OS'])
valdf = valdf.to_numpy()
val_tensor = torch.tensor(valdf, dtype=torch.float32)

pca = PCA(n_components=34)
X_pca = pca.fit_transform(X_df)
val_x_pca = pca.fit_transform(valdf)
PCA_survival_df = pd.DataFrame(X_pca)
PCA_survival_df['time'] = OStime.values
PCA_survival_df['event'] = OSevent.values
PCA_val_df = pd.DataFrame(val_x_pca)
PCA_val_df['time'] = val_OStime.values
PCA_val_df['event'] = val_event.values

def finetune_survival(model_path='best_autoencoder.pth'):
    data = pd.read_csv('multiomics.csv')
    surv = pd.read_csv('survival.csv')
    data = data[data['CA_type'] == 'COADREAD']
    X = pd.merge(data, surv, on='sample').dropna()
    durations = torch.tensor(X['OS.time'].to_numpy(), dtype=torch.float32)
    events = torch.tensor(X['OS'].to_numpy(), dtype=torch.float32)
    X_tensor = torch.tensor(X.drop(columns=['sample', 'CA_type', 'OS.time', 'OS']).to_numpy(), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, durations, events)
    train_size = int(0.7 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    encoder = Autoencoder(X_tensor.shape[1]).to(device)
    encoder.load_state_dict(torch.load(model_path))
    survival_model = SurvivalPredictor(128).to(device)
    optimizer = optim.Adam(survival_model.parameters(), lr=1e-4)

    for epoch in range(50):
        survival_model.train()
        train_loss = 0
        for fea, dur, evt in train_loader:
            encode = encoder.encoder(fea.to(device))
            risk = survival_model(encode).squeeze()
            loss = cox_loss(risk, dur.to(device), evt.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1} survival training loss: {train_loss/len(train_loader):.4f}")

    torch.save(survival_model.state_dict(), 'finetuned_coadread_survival.pth')
    return survival_model




class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(23904, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(2000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(1000, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(256, 128)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(128, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(256, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(1000, 2000)
        self.bn3 = nn.BatchNorm1d(2000)
        self.relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(2000, 23904)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
MMOAE = Autoencoder()


MMOAE.load_state_dict(torch.load('best_autoencoder.pth', weights_only=True))

pretrain_encoded_feature = []
pretrain_encoded_labels = []
with torch.no_grad():
    fea = MMOAE.encoder(X_tensor)
pretrain_encoded_feature.append(fea)

pretrain_encoded_features = torch.cat(pretrain_encoded_feature, dim=0)
pretrain_X_encoded = pretrain_encoded_features.cpu().numpy()
pretrain_survival_df = pd.DataFrame(pretrain_X_encoded)
pretrain_survival_df['time'] = OStime.values
pretrain_survival_df['event'] = OSevent.values

val_pretrain_encoded_feature = []
with torch.no_grad():
    fea = MMOAE.encoder(val_tensor)
val_pretrain_encoded_feature.append(fea)
val_pretrain_encoded_feature = torch.cat(val_pretrain_encoded_feature, dim=0)
val_pretrain_X_encoded = val_pretrain_encoded_feature.cpu().numpy()
val_pretrain_df = pd.DataFrame(val_data)
val_pretrain_df['time'] = val_OStime.values
val_pretrain_df['event'] = val_event.values


MMOAE = Autoencoder()
MMOAE.load_state_dict(torch.load('finetuned_coadread_survival.pth', weights_only=True))

finetune_encoded_feature = []
finetune_encoded_labels = []
with torch.no_grad():
    fea = MMOAE.encoder(X_tensor)
finetune_encoded_feature.append(fea)

finetune_encoded_features = torch.cat(finetune_encoded_feature, dim=0)
finetune_X_encoded = finetune_encoded_features.cpu().numpy()
survival_df = pd.DataFrame(finetune_X_encoded)
survival_df['time'] = OStime.values
survival_df['event'] = OSevent.values


val_encoded_feature = []
with torch.no_grad():
    fea = MMOAE.encoder(val_tensor)
val_encoded_feature.append(fea)
val_encoded_feature = torch.cat(val_encoded_feature, dim=0)
val_data = val_encoded_feature.cpu().numpy()
val_df = pd.DataFrame(val_data)
val_df['time'] = val_OStime.values
val_df['event'] = val_event.values

def cox_feature_selection(X_data, survival_df, alpha=0.05):
    feature_pvalues = {}
    significant_indices = []
    
    
    for i in range(X_data.shape[1]):
        feature_name = f'feature_{i}'
        
        
        feature_df = pd.DataFrame({
            feature_name: X_data[:, i],
            'time': survival_df['time'],
            'event': survival_df['event']
        })
        
        
        cph = CoxPHFitter()
        cph.fit(feature_df, duration_col='time', event_col='event')
        
        
        p_value = cph.summary.p.values[0]
        feature_pvalues[feature_name] = p_value
        
        
        if p_value < alpha:
            significant_indices.append(i)
            print(f"{feature_name}: p_value = {p_value:.4f} - significant")
        else:
            print(f"{feature_name}: p值 = {p_value:.4f} - no significant")
    
   
    if len(significant_indices) == 0:
        print("no significant features found.")
        return X_data, list(range(X_data.shape[1])), feature_pvalues
    
    
    significant_features = X_data[:, significant_indices]
    
    
    return significant_features, significant_indices


def find_optimal_clusters(X_data, max_clusters=10):
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        print(f"Evaluating {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_data)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.4f}")
    
    # Find the optimal number of clusters
    optimal_n_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2 clusters
    
    
    return optimal_n_clusters

# Step 2: Apply K-means with optimal number of clusters
def apply_kmeans(X_data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_data)
    return cluster_labels, kmeans

def KM_plot(survival_df, cluster_labels, c_index, c_index_std):
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    
    subtypes = survival_df['Cluster'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes)))

    for i, subtype in enumerate(subtypes):
        mask = survival_df['Cluster'] == subtype
        kmf.fit(
            survival_df.loc[mask, 'time'], 
            survival_df.loc[mask, 'event'], 
            label=f"{subtype} (n={sum(mask)})"
        )
        kmf.plot(ci_show=True, color=colors[i])
    
    plt.title('Kaplan-Meier survival curve', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=14)
    plt.ylabel('Survival probability', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    
    c_text = f"C-index = {c_index:.3f} ± {c_index_std:.3f}"
    plt.text(
        0.5, 0.02,  
        c_text,
        ha='center', va='center',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    
    plt.subplots_adjust(bottom=0.15)  
    
    plt.tight_layout()  
    plt.savefig('COADREAD/finetune_Kaplan-Meier_curve.tif', dpi=300, bbox_inches='tight')


    
def log_ranktest(survival_df, cluster_labels):
    
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    
    from lifelines.statistics import multivariate_logrank_test
    
    
    results = multivariate_logrank_test(
        survival_df['time'], 
        survival_df['Cluster'], 
        survival_df['event']
    )
    
    
    
    
    
    results_table = []    
    results_table.append({
    'Statistics': results.test_statistic,
    'p_value': results.p_value})
        


    logrank_results = pd.DataFrame(results_table)
    logrank_results.to_csv('COADREAD/finetune_logrank_test_results.csv', index=False, encoding='utf_8_sig')
    return results_table



def cv_cindex(significant_features, cv=5):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_c_indices = []
    significant_features = pd.DataFrame(significant_features)
    significant_features['time'] = OStime.values
    significant_features['event'] = OSevent.values
    for train_idx, test_idx in kf.split(significant_features):
        train_data = significant_features.iloc[train_idx]
        test_data = significant_features.iloc[test_idx]
    
        
        cv_cph = CoxPHFitter()
        cv_cph.fit(train_data, duration_col='time', event_col='event')
    
    
        pred = cv_cph.predict_partial_hazard(test_data)
        c_idx = concordance_index(test_data['time'], -pred, test_data['event'])
        cv_c_indices.append(c_idx)

    print(f"5_cv_C-index: {np.mean(cv_c_indices):.4f} (SDerror: {np.std(cv_c_indices):.4f})")
    return np.mean(cv_c_indices), np.std(cv_c_indices)


    
   

def main(X_encoded, survival_df, max_clusters=10):
    significant_features, significant_indices = cox_feature_selection(X_encoded, survival_df)
    n_clusters = find_optimal_clusters(significant_features, max_clusters)
    
    cluster_labels, kmeans = apply_kmeans(significant_features, n_clusters)
    
    cindex_mean, cindex_std = cv_cindex(significant_features, cv=5)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    for cluster, count in zip(unique_clusters, counts):
        
    KM_plot(survival_df, cluster_labels,cindex_mean, cindex_std)
    
    table = log_ranktest(survival_df, cluster_labels)
    
    
    return n_clusters, table

opti_clusters, logranktable = main(finetune_X_encoded, survival_df,
                                            max_clusters=10)


def pretrain_KM_plot(survival_df, cluster_labels, c_index, c_index_std):
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    
    subtypes = survival_df['Cluster'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes)))

    for i, subtype in enumerate(subtypes):
        mask = survival_df['Cluster'] == subtype
        kmf.fit(
            survival_df.loc[mask, 'time'], 
            survival_df.loc[mask, 'event'], 
            label=f"{subtype} (n={sum(mask)})"
        )
        kmf.plot(ci_show=True, color=colors[i])
    
    plt.title('Kaplan-Meier survival curve', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=14)
    plt.ylabel('Survival probability', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    
    c_text = f"C-index = {c_index:.3f} ± {c_index_std:.3f}"
    plt.text(
        0.5, 0.02,  
        c_text,
        ha='center', va='center',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    
    plt.subplots_adjust(bottom=0.15)  
    
    plt.tight_layout()  
    plt.savefig('COADREAD/pretrain_Kaplan-Meier_curve.tif', dpi=300, bbox_inches='tight')



def pretrain_log_ranktest(survival_df, cluster_labels):
    
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    
    from lifelines.statistics import multivariate_logrank_test
    
   
    results = multivariate_logrank_test(
        survival_df['time'], 
        survival_df['Cluster'], 
        survival_df['event']
    )
    
    
    
    results_table = []    
    results_table.append({
    'Statistics': results.test_statistic,
    'p_value': results.p_value})
        


    logrank_results = pd.DataFrame(results_table)
    logrank_results.to_csv('COADREAD/pretrain_logrank_test_results.csv', index=False, encoding='utf_8_sig')
    return results_table



def nonfinetune_compute(X_encoded, survival_df, max_clusters=10):
    significant_features, significant_indices = cox_feature_selection(X_encoded, survival_df)
    n_clusters = opti_clusters
   
    cluster_labels, kmeans = apply_kmeans(significant_features, n_clusters)
    
    cindex_mean, cindex_std = cv_cindex(significant_features, cv=5)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    for cluster, count in zip(unique_clusters, counts):
        
    pretrain_KM_plot(survival_df, cluster_labels,cindex_mean, cindex_std)
    
    table = pretrain_log_ranktest(survival_df, cluster_labels)
    
    
    return n_clusters, table

nclusters, logranktable = nonfinetune_compute(pretrain_X_encoded, pretrain_survival_df,
                                            max_clusters=10)




def PCA_KM_plot(survival_df, cluster_labels, c_index, c_index_std):
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(12, 8))
    
    subtypes = survival_df['Cluster'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(subtypes)))

    for i, subtype in enumerate(subtypes):
        mask = survival_df['Cluster'] == subtype
        kmf.fit(
            survival_df.loc[mask, 'time'], 
            survival_df.loc[mask, 'event'], 
            label=f"{subtype} (n={sum(mask)})"
        )
        kmf.plot(ci_show=True, color=colors[i])
    
    plt.title('Kaplan-Meier survival curve', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=14)
    plt.ylabel('Survival probability', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)

    
    c_text = f"C-index = {c_index:.3f} ± {c_index_std:.3f}"
    plt.text(
        0.5, 0.02,  
        c_text,
        ha='center', va='center',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8)
    )

    
    plt.subplots_adjust(bottom=0.15)  
    
    plt.tight_layout()  
    plt.savefig('COADREAD/PCA_Kaplan-Meier_curve.tif', dpi=300, bbox_inches='tight')



def PCA_log_ranktest(survival_df, cluster_labels):
    
    survival_df = survival_df.copy()
    survival_df['Cluster'] = cluster_labels
    
    
    from lifelines.statistics import multivariate_logrank_test
    
    
    results = multivariate_logrank_test(
        survival_df['time'], 
        survival_df['Cluster'], 
        survival_df['event']
    )
    
    
    
    results_table = []    
    results_table.append({
    'Statitics': results.test_statistic,
    'p_value': results.p_value})
        


    logrank_results = pd.DataFrame(results_table)
    logrank_results.to_csv('COADREAD/PCA_logrank_test_results.csv', index=False, encoding='utf_8_sig')
    return results_table



def PCA_compute(X_encoded, survival_df, max_clusters=10):
    significant_features, significant_indices = cox_feature_selection(X_encoded, survival_df)
    n_clusters = opti_clusters
    
    cluster_labels, kmeans = apply_kmeans(significant_features, n_clusters)
    
    cindex_mean, cindex_std = cv_cindex(significant_features, cv=5)
    
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    
    for cluster, count in zip(unique_clusters, counts):
        
    PCA_KM_plot(survival_df, cluster_labels,cindex_mean, cindex_std)
    
    table = PCA_log_ranktest(survival_df, cluster_labels)
    
    
    return n_clusters, table

nclusters, logranktable = PCA_compute(X_pca, PCA_survival_df,
                                            max_clusters=10)


