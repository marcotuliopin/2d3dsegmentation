import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def plot_loss_curves(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))

    plt.style.use('seaborn-v0_8')
    epochs = np.arange(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 
        color='#1f77b4',
        linewidth=2.5, 
        marker='o',
        markersize=8,
        label='Train Loss')
    
    plt.plot(epochs, val_losses, 
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


def visualize_predictions(model, data_loader, device, num_samples=5, save_path=None, rgb_only=True, use_hha=False):
    model.eval()
    batch = next(iter(data_loader))
    
    # Extract data based on model type
    if rgb_only:
        images, masks = batch
    elif use_hha:
        images, masks, hha = batch
        concat_inputs = torch.cat((images, hha), dim=1)  # Combine RGB and HHA
    else:
        images, masks, depth = batch
        concat_inputs = torch.cat((images, depth), dim=1)  # Combine RGB and depth
    
    # Limitar ao número disponível no batch
    batch_size = images.shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Prepare inputs for the model
    if rgb_only:
        model_input = images[:num_samples].to(device)
    else:
        model_input = concat_inputs[:num_samples].to(device)
        
    masks = masks[:num_samples].cpu().numpy()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(model_input)
        if isinstance(outputs, dict):
            outputs = outputs["out"]
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    
    # Denormalizar apenas as imagens RGB para visualização
    display_images = images[:num_samples].cpu().numpy()
    display_images = np.transpose(display_images, (0, 2, 3, 1))
    # Reversão da normalização para visualização
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    display_images = std * display_images + mean
    display_images = np.clip(display_images, 0, 1)

    # Carregar mapeamento id2label se fornecido
    # id2label_path = "data/nyuv2/id2label.json"
    # id2label = None
    # if id2label_path:
    #     try:
    #         import json
    #         with open(id2label_path, 'r') as f:
    #             id2label = json.load(f)
    #     except Exception as e:
    #         print(f"Error while loading id2label: {e}")
    
    # Coletar todas as classes presentes nas imagens para a legenda
    all_classes = set()
    for i in range(num_samples):
        all_classes.update(np.unique(masks[i]))
        all_classes.update(np.unique(preds[i]))
    all_classes = sorted(all_classes)
    
    colors = get_color_map()
    cmap = ListedColormap(colors)
    
    # Create image grid
    fig = plt.figure(figsize=(18, 4*num_samples))
    gs = fig.add_gridspec(num_samples, 4, width_ratios=[1, 1, 1, 0.5])
    
    # Create patches for legend
    legend_patches = []
    for class_id in all_classes:
        color = cmap(class_id)
        class_name = id2label[str(class_id)] if id2label and str(class_id) in id2label else f"Class {class_id}"
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, label=f"{class_id}: {class_name}"))
    
    # Adicionar visualizações
    for i in range(num_samples):
        # Imagem original - Use display_images que já está no formato correto
        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(display_images[i])
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Ground Truth - Usar colormap personalizado
        ax2 = fig.add_subplot(gs[i, 1])
        gt_img = ax2.imshow(masks[i], cmap=cmap, vmin=0, vmax=len(colors)-1)
        ax2.set_title("Ground Truth")
        ax2.axis('off')
        
        # Predição - Usar colormap personalizado
        ax3 = fig.add_subplot(gs[i, 2])
        pred_img = ax3.imshow(preds[i], cmap=cmap, vmin=0, vmax=len(colors)-1)
        ax3.set_title("Predicted Mask")
        ax3.axis('off')
        
        # Adicionar legenda apenas na primeira linha
        if i == 0:
            # Legenda à direita
            ax4 = fig.add_subplot(gs[:, 3])
            ax4.axis('off')
            ax4.legend(
                handles=legend_patches, 
                loc='center', 
                title="Classes",
                fontsize=8
            )
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for legend
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved at path: {save_path}")
    else:
        plt.show()
    
    
def get_color_map():
    colors = []

    tab20 = plt.cm.get_cmap('tab20', 20)
    colors.extend([tab20(i) for i in range(20)])
    
    tab20b = plt.cm.get_cmap('tab20b', 20)
    colors.extend([tab20b(i) for i in range(20)])
    
    tab20c = plt.cm.get_cmap('tab20c', 20)
    colors.extend([tab20c(i) for i in range(20)])

    colors[40] = (0.0, 0.0, 0.0, 1.0)

    return colors


def create_custom_cmap(colors):
    return LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(colors))
    
    
def visualize_image(image_path, title='Image', save_path=None):
    image = plt.imread(image_path)
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Image saved at path: {save_path}")
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