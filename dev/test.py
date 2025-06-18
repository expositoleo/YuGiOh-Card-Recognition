from multiprocessing.dummy import freeze_support
import os
import hashlib
import torch
from train import SiameseNetwork
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

model_path = "dev/trained.pth"

if __name__ == '__main__':
    # Verifica tamaño
    freeze_support()
    file_size = os.path.getsize(model_path)
    print(f"Tamaño del archivo: {file_size / (1024**2):.2f} MB")

    # Calcula MD5 (puede tardar en archivos grandes)
    print("Calculando hash MD5...")
    hash_md5 = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    print(f"MD5: {hash_md5.hexdigest()}")

    # Intenta cargarlo con torch
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✅ Archivo cargado CORRECTAMENTE")
    except Exception as e:
        print(f"❌ Error cargando modelo: {str(e)}")
        exit(1)

    # Configuración básica
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Instanciar el modelo
    model = SiameseNetwork().to(device)

    # Cargar pesos entrenados
    model.load_state_dict(checkpoint)
    model.eval()  # Modo evaluación (desactiva dropout/batch norm)
    print("✅ Modelo cargado y listo para usar")

    # Configurar transformaciones
    transform = transforms.Compose([
        transforms.Resize((255, 255)),
        transforms.ToTensor()
    ])

    def load_image(path):
        """Carga y transforma una imagen"""
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0)  # Añadir dimensión de batch

    def get_embedding(model, image_tensor):
        """Obtiene el embedding de una imagen"""
        with torch.no_grad():
            embedding = model.forward_once(image_tensor.to(device))
            return embedding

    def calculate_similarity(embedding1, embedding2):
        """Calcula la similitud coseno entre dos embeddings"""
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)
        return similarity.item()

    def find_most_similar_card(model, input_image_path, dataset_dir, top_k=5):
        """
        Encuentra las tarjetas más similares en el dataset
        
        Args:
            model: Modelo Siamese entrenado
            input_image_path: Ruta de la imagen de entrada
            dataset_dir: Directorio con las imágenes del dataset
            top_k: Número de resultados más similares a retornar
            
        Returns:
            Lista de tuplas (card_id, similarity_score)
        """
        # PASO 1: Procesar la imagen de entrada
        # - Cargar la imagen desde el archivo
        # - Aplicar las mismas transformaciones que se usaron durante el entrenamiento
        # - Obtener el embedding (vector de características) usando el modelo
        print(f"🔄 Procesando imagen de entrada: {os.path.basename(input_image_path)}")
        input_embedding = get_embedding(model, load_image(input_image_path))
        
        # PASO 2: Obtener lista de todas las imágenes del dataset
        # - Buscar todos los archivos .jpg en el directorio del dataset
        # - Estos son las tarjetas contra las que vamos a comparar
        dataset_images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
        print(f"📁 Encontradas {len(dataset_images)} tarjetas en el dataset")
        
        # PASO 3: Comparar la imagen de entrada con cada tarjeta del dataset
        similarities = []
        print(f"🔍 Comparando con {len(dataset_images)} tarjetas en el dataset...")
        
        for i, card_image in enumerate(dataset_images):
            try:
                # PASO 3.1: Procesar cada tarjeta del dataset
                # - Construir la ruta completa al archivo
                # - Cargar y transformar la imagen
                # - Obtener su embedding usando el mismo modelo
                card_path = os.path.join(dataset_dir, card_image)
                card_embedding = get_embedding(model, load_image(card_path))
                
                # PASO 3.2: Calcular similitud entre embeddings
                # - Usar similitud coseno: valores entre -1 y 1
                # - 1 = muy similar, 0 = no relacionado, -1 = muy diferente
                similarity = calculate_similarity(input_embedding, card_embedding)
                
                # PASO 3.3: Extraer ID de la tarjeta del nombre del archivo
                # - Los archivos se llaman "ID.jpg" (ej: "86100785.jpg")
                # - Removemos la extensión para obtener solo el ID
                card_id = card_image.replace('.jpg', '')
                
                # PASO 3.4: Guardar resultado
                similarities.append((card_id, similarity))
                
                # Mostrar progreso cada 100 tarjetas procesadas
                if (i + 1) % 100 == 0:
                    print(f"   Procesadas {i + 1}/{len(dataset_images)} tarjetas...")
                
            except Exception as e:
                print(f"❌ Error procesando {card_image}: {e}")
                continue
        
        # PASO 4: Ordenar resultados por similitud
        # - Ordenar de mayor a menor similitud (mejores coincidencias primero)
        # - El modelo Siamese fue entrenado para que imágenes similares tengan embeddings cercanos
        print("📊 Ordenando resultados por similitud...")
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # PASO 5: Retornar los top_k resultados más similares
        # - Retornar solo las mejores coincidencias
        # - Cada resultado es una tupla (card_id, similarity_score)
        return similarities[:top_k]

    # Probar con algunas imágenes del dataset
    images_dir = "D:/Facu/4to/Redes Neuro/YuGiOh-Card-Recognition/data/yugioh_card_images"
    
    # Seleccionar algunas imágenes para probar
    test_images = [
        "62121.jpg",
        "88472456.jpg", 
        "31339260.jpg",
        "47693640.jpg",
        "79852326.jpg"
    ]
    
    print("\n🧪 Probando el modelo con imágenes del dataset:")
    print("=" * 50)
    
    # Cargar embeddings de las imágenes de prueba
    embeddings = {}
    for img_name in test_images:
        img_path = os.path.join(images_dir, img_name)
        if os.path.exists(img_path):
            try:
                img_tensor = load_image(img_path)
                embedding = get_embedding(model, img_tensor)
                embeddings[img_name] = embedding
                print(f"✅ {img_name} procesada")
            except Exception as e:
                print(f"❌ Error procesando {img_name}: {e}")
        else:
            print(f"⚠️  {img_name} no encontrada")
    
    # Calcular similitudes entre pares de imágenes
    print("\n📊 Similitudes entre imágenes:")
    print("-" * 40)
    
    image_names = list(embeddings.keys())
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            img1_name = image_names[i]
            img2_name = image_names[j]
            
            similarity = calculate_similarity(
                embeddings[img1_name], 
                embeddings[img2_name]
            )
            
            print(f"{img1_name} vs {img2_name}: {similarity:.4f}")
    
    # Probar con la misma imagen (debería dar similitud = 1.0)
    if len(embeddings) > 0:
        first_img = list(embeddings.keys())[0]
        same_similarity = calculate_similarity(
            embeddings[first_img], 
            embeddings[first_img]
        )
        print(f"\n🔍 Similitud de {first_img} consigo misma: {same_similarity:.4f}")
    
    print("\n✅ Prueba completada exitosamente!")
    
    # 🎯 DEMOSTRACIÓN: Encontrar la tarjeta más similar
    print("\n" + "="*60)
    print("🎯 DEMOSTRACIÓN: Búsqueda de tarjeta más similar")
    print("="*60)
    
    # Usar una imagen del dataset como ejemplo
    sample_image = os.path.join("D:/Facu/4to/Redes Neuro/YuGiOh-Card-Recognition/dev/snatchsteal.jpg")
    
    if os.path.exists(sample_image):
        print(f"🔍 Buscando tarjetas similares a: {sample_image}")
        print("-" * 50)
        
        # Encontrar las 5 tarjetas más similares
        similar_cards = find_most_similar_card(
            model=model,
            input_image_path=sample_image,
            dataset_dir=images_dir,
            top_k=5
        )
        
        print(f"\n📊 Top 5 tarjetas más similares:")
        print("-" * 30)
        for i, (card_id, similarity) in enumerate(similar_cards, 1):
            print(f"{i}. ID: {card_id} | Similitud: {similarity:.4f}")
            
        # La primera debería ser la misma imagen (similitud = 1.0)
        if similar_cards[0][1] > 0.99:
            print(f"\n✅ Verificación exitosa: La imagen más similar es la misma (similitud = {similar_cards[0][1]:.4f})")
        else:
            print(f"\n⚠️  La imagen más similar no es la misma (similitud = {similar_cards[0][1]:.4f})")
        
        plt.imshow(Image.open(sample_image))
        plt.title(f"Imagen de entrada")
        plt.axis('off')

        plt.figure()

        plt.imshow(Image.open(os.path.join(images_dir, f"{similar_cards[0][0]}.jpg")))
        plt.title(f"Imagen más similar: {similar_cards[0][0]}")
        plt.axis('off')

        plt.figure()

        plt.show()

    else:
        print(f"❌ No se encontró la imagen de ejemplo: {sample_image}")
    
    print("\n" + "="*60)
    print("💡 USO: Para encontrar la tarjeta más similar a cualquier imagen:")
    print("similar_cards = find_most_similar_card(model, 'ruta/a/tu/imagen.jpg', images_dir)")
    print("card_id = similar_cards[0][0]  # ID de la tarjeta más similar")
    print("="*60)

