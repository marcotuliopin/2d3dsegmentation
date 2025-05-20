import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))

    plt.style.use('seaborn-v0_8')
    epochs = np.arange(1, len(train_losses) + 1)
    
    train_line, = plt.plot(epochs, train_losses, 
                          color='#1f77b4',
                          linewidth=2.5, 
                          marker='o',
                          markersize=8,
                          label='Train Loss')
    
    val_line, = plt.plot(epochs, val_losses, 
                        color='#ff7f0e',
                        linewidth=2.5, 
                        marker='s',
                        markersize=8,
                        label='Validation Loss')
    
    # Ajustes estéticos
    plt.title('Training and Validation Loss', fontsize=14, pad=20)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)
    
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    plt.annotate(f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}',
                xy=(0.98, 0.98),
                xycoords='axes fraction',
                ha='right',
                va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend(fontsize=12, framealpha=1)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot salvo em: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, normalize=True, save_path=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {save_path}")
    else:
        plt.show()


def visualize_predictions(model, data_loader, device, num_samples=5, save_path=None):
    model.eval()
    images, masks = next(iter(data_loader))
    
    # Limitar ao número disponível no batch
    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)
    
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].cpu().numpy()
    
    with torch.no_grad():
        outputs = model(images)['out']
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Denormalizar imagens
    images = images.cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    # Reversão da normalização para visualização
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    images = std * images + mean
    images = np.clip(images, 0, 1)
    
    plt.figure(figsize=(15, 4*num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i*3+1)
        plt.imshow(images[i])
        plt.title("Imagem Original")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+2)
        plt.imshow(masks[i], cmap='tab20')
        plt.title("Máscara Real")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i*3+3)
        plt.imshow(preds[i], cmap='tab20')
        plt.title("Predição")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualizações salvas em: {save_path}")
    else:
        plt.show()
    
    
def visualize_image(image_path, title='Image', save_path=None):
    image = plt.imread(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Imagem salva em: {save_path}")
    else:
        plt.show()


def print_pair_image_label(image, label) -> None:
    unique_classes = np.unique(np.array(label))
    
    print("Image shape:", np.array(image).shape)
    print("Label shape:", np.array(label).shape)
    
    # Configurar a figura
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Imagem RGB
    axes[0].imshow(image)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')
    
    # Imagem de segmentação
    seg_img = axes[1].imshow(label, cmap="tab20")
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Criar legenda com cores para cada classe
    cmap = plt.cm.tab20
    norm = plt.Normalize(vmin=0, vmax=19)  # tab20 tem 20 cores
    
    # Criar patches para a legenda
    legend_patches = []
    for class_id in unique_classes:
        color = cmap(norm(class_id % 20))  # Usar módulo 20 para evitar erros se houver mais de 20 classes
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f"Class {class_id}"))
    
    # Adicionar legenda fora da imagem
    fig.legend(
        handles=legend_patches,
        loc='center right', 
        title="Classes",
        bbox_to_anchor=(1.1, 0.5),
        fontsize=8
    )
    
    plt.tight_layout()
    plt.show()