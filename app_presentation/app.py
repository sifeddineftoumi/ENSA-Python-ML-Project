"""
Application Web Interactive Complète - Projet Python ENSA Berrechid 2025-2026
Version Finale avec Toutes les Fonctionnalités
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Projet ML ENSA - Interactive",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_resource
def load_mnist_model():
    """Charger le modèle CNN MNIST pré-entraîné"""
    try:
        from tensorflow import keras
        model = keras.models.load_model('../partie1_cnn_mnist/outputs/mnist_cnn_model.h5')
        return model
    except:
        return None

def preprocess_image_for_mnist(image):
    """Prétraiter une image pour MNIST"""
    img = image.convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def generate_synthetic_sales_data(n_days, base_sales, growth_rate, 
                                  winter_factor, summer_factor, 
                                  autumn_factor, saturday_factor):
    """Générer des données de ventes synthétiques"""
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_days, freq='D')
    
    trend = np.linspace(base_sales, base_sales * (1 + growth_rate), n_days)
    
    seasonal_annual = np.array([
        winter_factor if m in [12, 1, 2] 
        else 1.0 if m in [3, 4, 5]
        else summer_factor if m in [6, 7, 8]
        else autumn_factor
        for m in [d.month for d in dates]
    ])
    
    seasonal_weekly = np.array([
        saturday_factor if d.weekday() == 5 else 1.0
        for d in dates
    ])
    
    noise = np.random.normal(0, base_sales * 0.05, n_days)
    sales = trend * seasonal_annual * seasonal_weekly + noise
    sales = np.maximum(sales, 10).astype(int)
    
    return pd.DataFrame({'date': dates, 'sales': sales})

# ============================================================================
# SIDEBAR - NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisir une section :",
    [
        "🏠 Accueil",
        "🧠 CNN - Entraîner & Tester",
        "📊 Ventes - Charger Données",
        "🤖 Ventes - Entraîner Modèles",
        "🔮 Ventes - Prédictions"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Projet réalisé par :**  
ENSA Berrechid  
Année 2025/2026

**Technologies :**  
• Python • Streamlit  
• TensorFlow • Prophet  
• SARIMA • Plotly
""")

# ============================================================================
# PAGE 1 : ACCUEIL
# ============================================================================

if page == "🏠 Accueil":
    st.markdown('<div class="main-title">🤖 Application ML Interactive</div>', 
                unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #555;">Entraînement, Test et Prédiction en Direct</h2>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Fonctionnalités Disponibles
        
        #### 🧠 Partie 1 : CNN sur MNIST
        - ✅ **Entraîner** un CNN depuis l'interface
        - ✅ **Tester** avec des chiffres MNIST
        - ✅ **Uploader** vos propres images
        - ✅ **Visualiser** les performances en temps réel
        
        #### 📊 Partie 2 : Prédiction des Ventes
        - ✅ **Charger** votre propre fichier CSV
        - ✅ **Générer** des données synthétiques
        - ✅ **Entraîner** SARIMA et Prophet
        - ✅ **Comparer** les modèles automatiquement
        - ✅ **Prédire** les 30 prochains jours
        - ✅ **Télécharger** les résultats (CSV)
        """)
    
    with col2:
        st.success("""
        ### 🚀 Comment Utiliser l'Application
        
        **Étape 1 :** Choisissez une section dans le menu à gauche
        
        **Étape 2 :** Suivez les instructions interactives
        
        **Étape 3 :** Entraînez, testez, prédisez !
        
        **Étape 4 :** Téléchargez vos résultats
        """)
        
        st.info("""
        ### 💡 Nouveautés
        
        - 🎨 Interface moderne et intuitive
        - 📤 Upload de fichiers CSV/images
        - ⚡ Entraînement en temps réel
        - 📊 Visualisations interactives
        - 💾 Export des résultats
        """)
    
    st.markdown("---")
    
    # Instructions rapides
    st.markdown("### 📝 Guide Rapide")
    
    tab1, tab2 = st.tabs(["🧠 CNN MNIST", "📊 Prédiction Ventes"])
    
    with tab1:
        st.markdown("""
        **Pour tester le CNN :**
        
        1. Allez dans **"🧠 CNN - Entraîner & Tester"**
        2. Choisissez l'onglet **"🎨 Tester avec MNIST"**
        3. Cliquez sur **"Générer un Chiffre"**
        4. Voyez la prédiction instantanément !
        
        **Ou uploadez une image :**
        
        1. Onglet **"📤 Upload Image"**
        2. Glissez votre image (PNG/JPG)
        3. Prédiction automatique !
        """)
    
    with tab2:
        st.markdown("""
        **Pour prédire les ventes :**
        
        1. Allez dans **"📊 Ventes - Charger Données"**
        2. **Option A :** Générez des données synthétiques
        3. **Option B :** Uploadez votre fichier CSV
        4. Allez dans **"🤖 Ventes - Entraîner Modèles"**
        5. Entraînez SARIMA et Prophet
        6. Comparez les performances
        7. Allez dans **"🔮 Ventes - Prédictions"**
        8. Obtenez les prédictions futures
        9. Téléchargez les résultats !
        """)

# ============================================================================
# PAGE 2 : CNN - ENTRAÎNER & TESTER
# ============================================================================

elif page == "🧠 CNN - Entraîner & Tester":
    st.markdown('<div class="main-title">🧠 Classification de Chiffres - CNN</div>', 
                unsafe_allow_html=True)
    
    tabs = st.tabs(["🏋️ Entraîner CNN", "🎨 Tester avec MNIST", "📤 Upload Image", "📊 Résultats"])
    
    # ========================================================================
    # TAB 1 : ENTRAÎNEMENT
    # ========================================================================
    
    with tabs[0]:
        st.markdown("### 🏋️ Entraîner un Modèle CNN sur MNIST")
        
        st.warning("""
        ⚠️ **Note importante :** L'entraînement d'un CNN peut prendre 2-5 minutes.
        Pour une démo rapide, utilisez plutôt les onglets "Tester avec MNIST" ou "Upload Image".
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Configuration du modèle :**")
            
            epochs = st.slider("Nombre d'epochs", min_value=1, max_value=20, value=5)
            batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=2)
            
            if st.button("🚀 Lancer l'Entraînement", type="primary"):
                with st.spinner("Entraînement en cours... Cela peut prendre quelques minutes."):
                    try:
                        from tensorflow import keras
                        from keras import layers
                        
                        # Charger MNIST
                        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
                        
                        # Prétraitement
                        X_train = X_train.astype('float32') / 255.0
                        X_test = X_test.astype('float32') / 255.0
                        X_train = np.expand_dims(X_train, axis=-1)
                        X_test = np.expand_dims(X_test, axis=-1)
                        y_train_cat = keras.utils.to_categorical(y_train, 10)
                        y_test_cat = keras.utils.to_categorical(y_test, 10)
                        
                        # Créer le modèle
                        model = keras.Sequential([
                            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(64, (3, 3), activation='relu'),
                            layers.MaxPooling2D((2, 2)),
                            layers.Conv2D(128, (3, 3), activation='relu'),
                            layers.Flatten(),
                            layers.Dense(128, activation='relu'),
                            layers.Dropout(0.5),
                            layers.Dense(10, activation='softmax')
                        ])
                        
                        model.compile(optimizer='adam',
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy'])
                        
                        # Entraîner avec barre de progression
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        class StreamlitCallback(keras.callbacks.Callback):
                            def on_epoch_end(self, epoch, logs=None):
                                progress = (epoch + 1) / epochs
                                progress_bar.progress(progress)
                                status_text.text(f"Epoch {epoch+1}/{epochs} - "
                                               f"Loss: {logs['loss']:.4f} - "
                                               f"Accuracy: {logs['accuracy']:.4f}")
                        
                        history = model.fit(X_train, y_train_cat,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_split=0.1,
                                          callbacks=[StreamlitCallback()],
                                          verbose=0)
                        
                        # Évaluer
                        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
                        
                        # Sauvegarder dans session_state
                        st.session_state.cnn_model = model
                        st.session_state.cnn_history = history.history
                        st.session_state.cnn_test_acc = test_accuracy
                        
                        st.success(f"""
                        ✅ **Entraînement terminé !**
                        
                        - Accuracy finale : **{test_accuracy*100:.2f}%**
                        - Loss finale : **{test_loss:.4f}**
                        
                        Le modèle est maintenant prêt à être utilisé dans les autres onglets !
                        """)
                        
                        # Afficher les courbes
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
                        axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
                        axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
                        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
                        axes[0].set_xlabel('Epoch')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        axes[1].plot(history.history['loss'], label='Train', linewidth=2)
                        axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
                        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
                        axes[1].set_xlabel('Epoch')
                        axes[1].legend()
                        axes[1].grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")
        
        with col2:
            st.info("""
            **💡 Conseils :**
            
            - **5 epochs** : Rapide (~2 min)
            - **10 epochs** : Recommandé (~4 min)
            - **20 epochs** : Maximum (~8 min)
            
            **Batch size :**
            - Plus petit = plus précis
            - Plus grand = plus rapide
            """)
    
    # ========================================================================
    # TAB 2 : TESTER AVEC MNIST
    # ========================================================================
    
    with tabs[1]:
        st.markdown("### 🎨 Tester avec des Chiffres MNIST")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### ✏️ Générer un Chiffre")
            
            if st.button("🎲 Générer un Chiffre Aléatoire", type="primary"):
                try:
                    from tensorflow import keras
                    (_, _), (X_test, y_test) = keras.datasets.mnist.load_data()
                    
                    idx = np.random.randint(0, len(X_test))
                    test_img = X_test[idx]
                    true_label = y_test[idx]
                    
                    st.session_state.test_image = test_img
                    st.session_state.true_label = true_label
                    
                except:
                    test_img = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
                    st.session_state.test_image = test_img
                    st.session_state.true_label = None
        
        with col2:
            st.markdown("#### 📊 Prédiction")
            
            if 'test_image' in st.session_state:
                # Afficher l'image
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(st.session_state.test_image, cmap='gray')
                ax.axis('off')
                if st.session_state.true_label is not None:
                    ax.set_title(f'Vrai Label: {st.session_state.true_label}', fontsize=16)
                st.pyplot(fig)
                
                if st.button("🎯 Prédire", type="primary"):
                    if 'cnn_model' in st.session_state:
                        model = st.session_state.cnn_model
                    else:
                        model = load_mnist_model()
                    
                    if model:
                        img_array = st.session_state.test_image.astype('float32') / 255.0
                        img_array = img_array.reshape(1, 28, 28, 1)
                        
                        predictions = model.predict(img_array, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = np.max(predictions[0]) * 100
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 2rem; border-radius: 15px; color: white; text-align: center;">
                            <h1 style="font-size: 5rem; margin: 0;">{predicted_class}</h1>
                            <p style="font-size: 1.5rem;">Confiance : {confidence:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 📊 Distribution des Probabilités")
                        fig = px.bar(x=list(range(10)), y=predictions[0],
                                   labels={'x': 'Chiffre', 'y': 'Probabilité'},
                                   color=predictions[0],
                                   color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("❌ Modèle non disponible. Entraînez-le d'abord.")
    
    # ========================================================================
    # TAB 3 : UPLOAD IMAGE
    # ========================================================================
    
    with tabs[2]:
        st.markdown("### 📤 Upload une Image de Chiffre")
        
        uploaded_file = st.file_uploader("Choisissez une image (PNG, JPG)", 
                                        type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption='Image Originale', use_column_width=True)
            
            with col2:
                st.markdown("### 🔍 Traitement et Prédiction")
                
                img_processed = preprocess_image_for_mnist(image)
                
                st.image(img_processed.reshape(28, 28), 
                        caption='Image Traitée (28x28)', 
                        width=200,
                        clamp=True)
                
                if 'cnn_model' in st.session_state:
                    model = st.session_state.cnn_model
                else:
                    model = load_mnist_model()
                
                if model:
                    predictions = model.predict(img_processed, verbose=0)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0]) * 100
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-top: 2rem;">
                        <h1 style="font-size: 4rem; margin: 0;">{predicted_class}</h1>
                        <p style="font-size: 1.3rem;">Confiance : {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### 📊 Distribution des Probabilités")
                    fig = go.Figure(data=[
                        go.Bar(x=list(range(10)), y=predictions[0],
                              marker_color='lightblue')
                    ])
                    fig.update_layout(
                        xaxis_title="Chiffre",
                        yaxis_title="Probabilité",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ Modèle non chargé.")
    
    # ========================================================================
    # TAB 4 : RÉSULTATS
    # ========================================================================
    
    with tabs[3]:
        st.markdown("### 📊 Résultats de l'Entraînement")
        
        if 'cnn_history' in st.session_state:
            history = st.session_state.cnn_history
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Accuracy Finale", 
                         f"{st.session_state.cnn_test_acc*100:.2f}%")
            with col2:
                st.metric("📈 Epochs Entraînés", 
                         len(history['accuracy']))
            with col3:
                final_loss = history['loss'][-1]
                st.metric("📉 Loss Finale", 
                         f"{final_loss:.4f}")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(history['accuracy'], label='Train', linewidth=2)
            axes[0].plot(history['val_accuracy'], label='Validation', linewidth=2)
            axes[0].set_title('Evolution de l\'Accuracy', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(history['loss'], label='Train', linewidth=2)
            axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
            axes[1].set_title('Evolution de la Loss', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("ℹ️ Aucun modèle entraîné. Allez dans l'onglet 'Entraîner CNN'.")

# ============================================================================
# PAGE 3 : VENTES - CHARGER DONNÉES
# ============================================================================

elif page == "📊 Ventes - Charger Données":
    st.markdown('<div class="main-title">📊 Charger les Données de Ventes</div>', 
                unsafe_allow_html=True)
    
    tabs = st.tabs(["📤 Upload CSV", "🎲 Générer Données", "📊 Visualiser"])
    
    # ========================================================================
    # TAB 1 : UPLOAD CSV
    # ========================================================================
    
    with tabs[0]:
        st.markdown("### 📤 Upload votre Fichier CSV")
        
        st.info("""
        **Format attendu du CSV :**
        
        Le fichier doit contenir au minimum 2 colonnes :
        - `date` : Date au format YYYY-MM-DD
        - `sales` ou `ventes` : Nombre de ventes
        
        Exemple :
        ```
        date,sales
        2022-01-01,120
        2022-01-02,135
        ```
        """)
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, parse_dates=['date'])
                
                if 'date' not in df.columns:
                    st.error("❌ Le fichier doit contenir une colonne 'date'")
                elif 'sales' not in df.columns and 'ventes' not in df.columns:
                    st.error("❌ Le fichier doit contenir une colonne 'sales' ou 'ventes'")
                else:
                    if 'ventes' in df.columns:
                        df.rename(columns={'ventes': 'sales'}, inplace=True)
                    
                    st.session_state.sales_data = df
                    
                    st.success(f"✅ Fichier chargé ! **{len(df)} lignes** détectées.")
                    
                    st.markdown("#### 👀 Aperçu des Données")
                    st.dataframe(df.head(10))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("📊 Lignes", len(df))
                    with col2:
                        st.metric("📅 Première Date", str(df['date'].min().date()))
                    with col3:
                        st.metric("📅 Dernière Date", str(df['date'].max().date()))
                    with col4:
                        st.metric("📈 Moyenne", f"{df['sales'].mean():.1f}")
                    
                    fig = px.line(df, x='date', y='sales', title='Ventes dans le Temps')
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur : {str(e)}")
    
    # ========================================================================
    # TAB 2 : GÉNÉRER DONNÉES
    # ========================================================================
    
    with tabs[1]:
        st.markdown("### 🎲 Générer des Données Synthétiques")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            n_days = st.slider("Nombre de jours", 180, 1095, 730)
            base_sales = st.number_input("Ventes de base", 50, 200, 100)
            growth_rate = st.slider("Taux de croissance (%)", 0, 50, 15) / 100
            
            st.markdown("**Saisonnalité :**")
            col_a, col_b = st.columns(2)
            with col_a:
                winter_factor = st.slider("Hiver (boost %)", 0, 50, 30) / 100 + 1
                autumn_factor = st.slider("Automne (boost %)", 0, 50, 40) / 100 + 1
            with col_b:
                summer_factor = st.slider("Été (pénalité %)", -50, 0, -20) / 100 + 1
                saturday_factor = st.slider("Samedi (boost %)", 0, 100, 50) / 100 + 1
        
        with col2:
            st.info("""
            **💡 Conseils :**
            
            - **730 jours** = 2 ans
            - **Automne** : Rentrée
            - **Hiver** : Fêtes
            - **Samedi** : Shopping
            """)
        
        if st.button("🎲 Générer les Données", type="primary"):
            with st.spinner("Génération en cours..."):
                df = generate_synthetic_sales_data(
                    n_days, base_sales, growth_rate,
                    winter_factor, summer_factor,
                    autumn_factor, saturday_factor
                )
                
                st.session_state.sales_data = df
                
                st.success(f"✅ {n_days} jours générés !")
                
                st.dataframe(df.head(10))
                
                fig = px.line(df, x='date', y='sales', title='Données Générées')
                st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 3 : VISUALISER
    # ========================================================================
    
    with tabs[2]:
        st.markdown("### 📊 Visualiser les Données")
        
        if 'sales_data' not in st.session_state:
            st.warning("⚠️ Aucune donnée chargée. Uploadez un CSV ou générez des données.")
        else:
            df = st.session_state.sales_data
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Moyenne", f"{df['sales'].mean():.1f}")
            with col2:
                st.metric("📈 Maximum", f"{df['sales'].max()}")
            with col3:
                st.metric("📉 Minimum", f"{df['sales'].min()}")
            with col4:
                st.metric("📏 Écart-type", f"{df['sales'].std():.1f}")
            
            st.markdown("#### 📈 Série Temporelle")
            fig = px.line(df, x='date', y='sales',
                         title='Ventes Quotidiennes',
                         labels={'sales': 'Ventes', 'date': 'Date'})
            fig.update_traces(line_color='#1f77b4', line_width=2)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 Distribution")
            fig = px.histogram(df, x='sales', nbins=30, title='Distribution des Ventes')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 4 : VENTES - ENTRAÎNER MODÈLES
# ============================================================================

elif page == "🤖 Ventes - Entraîner Modèles":
    st.markdown('<div class="main-title">🤖 Entraîner les Modèles</div>', 
                unsafe_allow_html=True)
    
    if 'sales_data' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord charger des données")
    else:
        df = st.session_state.sales_data
        
        st.success(f"✅ Données chargées : {len(df)} jours")
        
        tabs = st.tabs(["⚙️ Configuration", "🔧 SARIMA", "🔮 Prophet", "📊 Comparaison"])
        
        # ====================================================================
        # TAB 1 : CONFIGURATION
        # ====================================================================
        
        with tabs[0]:
            st.markdown("### ⚙️ Configuration du Split")
            
            train_ratio = st.slider("Ratio Train/Test", 0.6, 0.9, 0.8, 0.05)
            
            split_point = int(len(df) * train_ratio)
            train_df = df.iloc[:split_point]
            test_df = df.iloc[split_point:]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **📊 Train Set**
                - Taille : {len(train_df)} jours
                - Période : {str(train_df['date'].min().date())} → {str(train_df['date'].max().date())}
                """)
            
            with col2:
                st.info(f"""
                **📊 Test Set**
                - Taille : {len(test_df)} jours
                - Période : {str(test_df['date'].min().date())} → {str(test_df['date'].max().date())}
                """)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train_df['date'], y=train_df['sales'],
                                    mode='lines', name='Train', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=test_df['date'], y=test_df['sales'],
                                    mode='lines', name='Test', line=dict(color='red')))
            fig.update_layout(title='Découpage Train/Test', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("💾 Sauvegarder le Split", type="primary"):
                st.session_state.train_data = train_df
                st.session_state.test_data = test_df
                st.success("✅ Split sauvegardé !")
        
        # ====================================================================
        # TAB 2 : SARIMA
        # ====================================================================
        
        with tabs[1]:
            st.markdown("### 🔧 Entraîner SARIMA")
            
            if 'train_data' not in st.session_state:
                st.warning("⚠️ Veuillez sauvegarder le split")
            else:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Paramètres SARIMA :**")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        p = st.number_input("p (AR)", 0, 5, 1)
                        d = st.number_input("d (I)", 0, 2, 1)
                        q = st.number_input("q (MA)", 0, 5, 1)
                    with col_b:
                        P = st.number_input("P (Seasonal)", 0, 2, 1)
                        D = st.number_input("D (Seasonal)", 0, 2, 1)
                        Q = st.number_input("Q (Seasonal)", 0, 2, 1)
                        s = st.number_input("s (Period)", 1, 365, 7)
                
                with col2:
                    st.info("""
                    **💡 Conseils :**
                    
                    - p, d, q : 1-2
                    - s=7 : Hebdo
                    - s=30 : Mensuel
                    """)
                
                if st.button("🚀 Entraîner SARIMA", type="primary"):
                    with st.spinner("Entraînement..."):
                        try:
                            from statsmodels.tsa.statespace.sarimax import SARIMAX
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            model = SARIMAX(
                                st.session_state.train_data['sales'],
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, s),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            model_fit = model.fit(disp=False)
                            
                            test_data = st.session_state.test_data
                            predictions = model_fit.forecast(steps=len(test_data))
                            
                            mse = mean_squared_error(test_data['sales'], predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(test_data['sales'], predictions)
                            mape = np.mean(np.abs((test_data['sales'] - predictions) / test_data['sales'])) * 100
                            r2 = r2_score(test_data['sales'], predictions)
                            
                            st.session_state.sarima_model = model_fit
                            st.session_state.sarima_predictions = predictions
                            st.session_state.sarima_metrics = {
                                'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
                            }
                            
                            st.success("✅ SARIMA entraîné !")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("RMSE", f"{rmse:.2f}")
                            with col2:
                                st.metric("MAE", f"{mae:.2f}")
                            with col3:
                                st.metric("MAPE", f"{mape:.2f}%")
                            with col4:
                                st.metric("R²", f"{r2:.4f}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['sales'],
                                                    mode='lines+markers', name='Réel'))
                            fig.add_trace(go.Scatter(x=test_data['date'], y=predictions,
                                                    mode='lines+markers', name='SARIMA'))
                            fig.update_layout(title='SARIMA - Prédictions vs Réalité', height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur : {str(e)}")
        
        # ====================================================================
        # TAB 3 : PROPHET
        # ====================================================================
        
        with tabs[2]:
            st.markdown("### 🔮 Entraîner Prophet")
            
            if 'train_data' not in st.session_state:
                st.warning("⚠️ Veuillez sauvegarder le split")
            else:
                if st.button("🚀 Entraîner Prophet", type="primary"):
                    with st.spinner("Entraînement..."):
                        try:
                            from prophet import Prophet
                            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                            
                            train_data = st.session_state.train_data.copy()
                            test_data = st.session_state.test_data.copy()
                            
                            train_prophet = train_data[['date', 'sales']].rename(
                                columns={'date': 'ds', 'sales': 'y'}
                            )
                            
                            model = Prophet(
                                yearly_seasonality=True,
                                weekly_seasonality=True,
                                daily_seasonality=False,
                                seasonality_mode='multiplicative'
                            )
                            model.fit(train_prophet)
                            
                            future = model.make_future_dataframe(periods=len(test_data), freq='D')
                            forecast = model.predict(future)
                            
                            predictions = forecast[forecast['ds'].isin(test_data['date'])]['yhat'].values
                            
                            mse = mean_squared_error(test_data['sales'], predictions)
                            rmse = np.sqrt(mse)
                            mae = mean_absolute_error(test_data['sales'], predictions)
                            mape = np.mean(np.abs((test_data['sales'] - predictions) / test_data['sales'])) * 100
                            r2 = r2_score(test_data['sales'], predictions)
                            
                            st.session_state.prophet_model = model
                            st.session_state.prophet_predictions = predictions
                            st.session_state.prophet_metrics = {
                                'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2
                            }
                            
                            st.success("✅ Prophet entraîné !")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("RMSE", f"{rmse:.2f}")
                            with col2:
                                st.metric("MAE", f"{mae:.2f}")
                            with col3:
                                st.metric("MAPE", f"{mape:.2f}%")
                            with col4:
                                st.metric("R²", f"{r2:.4f}")
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=test_data['date'], y=test_data['sales'],
                                                    mode='lines+markers', name='Réel'))
                            fig.add_trace(go.Scatter(x=test_data['date'], y=predictions,
                                                    mode='lines+markers', name='Prophet'))
                            fig.update_layout(title='Prophet - Prédictions', height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Erreur : {str(e)}")
        
        # ====================================================================
        # TAB 4 : COMPARAISON
        # ====================================================================
        
        with tabs[3]:
            st.markdown("### 📊 Comparaison des Modèles")
            
            if 'sarima_metrics' in st.session_state and 'prophet_metrics' in st.session_state:
                comparison_df = pd.DataFrame({
                    'Modèle': ['SARIMA', 'Prophet'],
                    'RMSE': [
                        st.session_state.sarima_metrics['RMSE'],
                        st.session_state.prophet_metrics['RMSE']
                    ],
                    'MAE': [
                        st.session_state.sarima_metrics['MAE'],
                        st.session_state.prophet_metrics['MAE']
                    ],
                    'MAPE (%)': [
                        st.session_state.sarima_metrics['MAPE'],
                        st.session_state.prophet_metrics['MAPE']
                    ],
                    'R²': [
                        st.session_state.sarima_metrics['R2'],
                        st.session_state.prophet_metrics['R2']
                    ]
                })
                
                st.dataframe(comparison_df, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(comparison_df, x='Modèle', y='RMSE',
                                color='RMSE', color_continuous_scale='Reds_r',
                                title='RMSE (plus bas = meilleur)')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(comparison_df, x='Modèle', y='R²',
                                color='R²', color_continuous_scale='Greens',
                                title='R² (plus haut = meilleur)')
                    st.plotly_chart(fig, use_container_width=True)
                
                best_model = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Modèle']
                st.success(f"🏆 **Meilleur Modèle : {best_model}**")
                
            else:
                st.info("ℹ️ Entraînez les 2 modèles pour voir la comparaison")

# ============================================================================
# PAGE 5 : VENTES - PRÉDICTIONS
# ============================================================================

elif page == "🔮 Ventes - Prédictions":
    st.markdown('<div class="main-title">🔮 Prédictions Futures</div>', 
                unsafe_allow_html=True)
    
    if 'sarima_model' not in st.session_state and 'prophet_model' not in st.session_state:
        st.warning("⚠️ Entraînez au moins un modèle")
    else:
        st.success("✅ Modèles disponibles")
        
        models_available = []
        if 'sarima_model' in st.session_state:
            models_available.append('SARIMA')
        if 'prophet_model' in st.session_state:
            models_available.append('Prophet')
        
        selected_model = st.selectbox("Choisissez un modèle", models_available)
        forecast_days = st.slider("Jours à prédire", 7, 90, 30)
        
        if st.button("🔮 Générer Prédictions", type="primary"):
            with st.spinner(f"Génération avec {selected_model}..."):
                try:
                    sales_data = st.session_state.sales_data
                    last_date = sales_data['date'].max()
                    future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                                periods=forecast_days, freq='D')
                    
                    if selected_model == 'SARIMA':
                        model = st.session_state.sarima_model
                        predictions = model.forecast(steps=forecast_days)
                        lower_bound = predictions - 1.96 * predictions.std()
                        upper_bound = predictions + 1.96 * predictions.std()
                    
                    elif selected_model == 'Prophet':
                        model = st.session_state.prophet_model
                        future = pd.DataFrame({'ds': future_dates})
                        forecast = model.predict(future)
                        predictions = forecast['yhat'].values
                        lower_bound = forecast['yhat_lower'].values
                        upper_bound = forecast['yhat_upper'].values
                    
                    future_df = pd.DataFrame({
                        'date': future_dates,
                        'prediction': predictions,
                        'lower': lower_bound,
                        'upper': upper_bound
                    })
                    
                    st.session_state.future_predictions = future_df
                    
                    st.success(f"✅ {forecast_days} jours prédits !")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📊 Moyenne", f"{predictions.mean():.1f}")
                    with col2:
                        st.metric("📈 Maximum", f"{predictions.max():.1f}")
                    with col3:
                        st.metric("📉 Minimum", f"{predictions.min():.1f}")
                    with col4:
                        st.metric("📏 Écart-type", f"{predictions.std():.1f}")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=sales_data['date'][-90:], 
                        y=sales_data['sales'][-90:],
                        mode='lines',
                        name='Historique',
                        line=dict(color='blue')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_df['date'],
                        y=future_df['prediction'],
                        mode='lines+markers',
                        name='Prédictions',
                        line=dict(color='red', width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_df['date'],
                        y=future_df['upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=future_df['date'],
                        y=future_df['lower'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(width=0),
                        name='Intervalle 95%'
                    ))
                    
                    fig.update_layout(
                        title=f'Prédictions - {forecast_days} Jours ({selected_model})',
                        xaxis_title='Date',
                        yaxis_title='Ventes',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("### 📋 Tableau des Prédictions")
                    st.dataframe(future_df.style.format({
                        'prediction': '{:.1f}',
                        'lower': '{:.1f}',
                        'upper': '{:.1f}'
                    }), use_container_width=True)
                    
                    csv = future_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger (CSV)",
                        data=csv,
                        file_name=f'predictions_{selected_model}_{forecast_days}days.csv',
                        mime='text/csv',
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")