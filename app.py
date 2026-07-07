import import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página web
st.set_page_config(page_title="Dashboard - Clustering de Textos", layout="wide")

# Carrega os dados exportados do pipeline
@st.cache_data
def carregar_dados():
    return joblib.load('dados_dashboard.pkl')

try:
    dados = carregar_dados()
except Exception as e:
    st.error(f"⚠️ Erro ao carregar o arquivo 'dados_dashboard.pkl': {e}")
    st.stop()

# Captura e trata o DataFrame de métricas (df_resultados)
if 'df_resultados' in dados:
    df_bruto = dados['df_resultados']
elif 'resultados_metricas' in dados:
    df_bruto = dados['resultados_metricas']
else:
    df_bruto = pd.DataFrame()

df_resultados = pd.DataFrame()
if not df_bruto.empty:
    df_temp = df_bruto.copy()
    df_temp.columns = [str(col).replace('Representacao', 'Representação') for col in df_temp.columns]
    df_resultados = df_temp

# =====================================================================
# BARRA LATERAL: CONFIGURAÇÕES E NAVEGAÇÃO COMPLETA
# =====================================================================
st.sidebar.header("Navigation & Settings")

# 1. Menu de Navegação Principal (Substitui o st.tabs com isolamento total)
tela_selecionada = st.sidebar.radio(
    "Selecione a Tela do Painel",
    ["📈 Desempenho e Métricas", "🔍 Exploração de Textos por Cluster"]
)

st.sidebar.markdown("---")

# 2. Seleção da Base de Dados
base_selecionada = st.sidebar.selectbox(
    "Escolha a Base de Dados", 
    ["Base 1 - Noticias", "Base 2 - 20Newsgroups"]
)


# =====================================================================
# TELA 1: DESEMPENHO E MÉTRICAS
# =====================================================================
if tela_selecionada == "📈 Desempenho e Métricas":
    st.title("📈 Desempenho e Métricas Gerais")
    st.subheader(f"Comparação de Desempenho Completo ({base_selecionada})")

    df_filtrado = pd.DataFrame()
    if not df_resultados.empty and 'Base' in df_resultados.columns:
        df_filtrado = df_resultados[df_resultados['Base'] == base_selecionada].copy()

    if not df_filtrado.empty:
        st.dataframe(df_filtrado)
        
        col_hue = 'Representação' if 'Representação' in df_filtrado.columns else df_filtrado.columns[1]
        st.markdown("### 📊 Visão Geral das Métricas de Clusterização")

        graf_col1, graf_col2 = st.columns(2)

        with graf_col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.barplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, palette='Set2', ax=ax1)
            ax1.set_ylabel("Silhouette Score (Maior é melhor)")
            ax1.set_title(f"Métrica Silhouette - {base_selecionada}")
            plt.tight_layout()
            st.pyplot(fig1, clear_figure=True)
            plt.close(fig1)

        with graf_col2:
            metricas_alternativas = [c for c in df_filtrado.columns if c in ['Davies-Bouldin', 'Davies_Bouldin', 'Calinski-Harabasz', 'Calinski_Harabasz', 'V-Measure', 'Completeness']]
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            if metricas_alternativas:
                metrica_alvo = metricas_alternativas[0]
                sns.barplot(data=df_filtrado, x='Algoritmo', y=metrica_alvo, hue=col_hue, palette='Set1', ax=ax2)
                ax2.set_ylabel(f"{metrica_alvo}")
                ax2.set_title(f"Métrica {metrica_alvo} - {base_selecionada}")
            else:
                sns.barplot(data=df_filtrado, x=col_hue, y='Silhouette', hue='Algoritmo', palette='Pastel1', ax=ax2)
                ax2.set_ylabel("Silhouette Score")
                ax2.set_title(f"Silhouette por Tipo de Representação - {base_selecionada}")
            plt.tight_layout()
            st.pyplot(fig2, clear_figure=True)
            plt.close(fig2)

        st.markdown("---")
        st.markdown("### 📈 Análise de Tendência de Desempenho")
        fig3, ax3 = plt.subplots(figsize=(10, 3.5))
        sns.pointplot(data=df_filtrado, x='Algoritmo', y='Silhouette', hue=col_hue, 
                      markers=["o", "s", "^"][:len(df_filtrado[col_hue].unique())], 
                      linestyles=["-", "--", "-."], ax=ax3)
        ax3.set_ylabel("Silhouette Score")
        ax3.set_title("Evolução do Coeficiente Silhouette por Combinação")
        ax3.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig3, clear_figure=True)
        plt.close(fig3)
    else:
        st.info("Nenhum dado de métrica encontrado para esta base.")


# =====================================================================
# TELA 2: EXPLORAÇÃO DE TEXTOS POR CLUSTER (TOTALMENTE INDEPENDENTE)
# =====================================================================
else:
    st.title("🔍 Exploração de Textos por Cluster")
    st.subheader(f"Documentos Agrupados na {base_selecionada}")

    chave_base = 'base_noticias' if base_selecionada == "Base 1 - Noticias" else 'base_newsgroups'

    dados_da_base = None
    if chave_base in dados:
        dados_da_base = dados[chave_base]
    elif 'dados_noticias_com_clusters' in dados and base_selecionada == "Base 1 - Noticias":
        dados_da_base = dados['dados_noticias_com_clusters']
    else:
        dados_da_base = dados.get(base_selecionada, None)

    if dados_da_base is None:
        st.error("⚠️ Estrutura de dados da base selecionada não encontrada no arquivo .pkl.")
    else:
        opcoes_disponiveis = [k for k in dados_da_base.keys() if k.startswith('labels_') and not k == 'labels']

        if len(opcoes_disponiveis) == 0:
            st.error("⚠️ Nenhuma coluna de cluster encontrada para exploração nesta base.")
        else:
            nomes_formatados = {k: k.replace('labels_kmeans_', 'KMeans + ').replace('labels_agglomerative_', 'Hierárquico + ').replace('labels_dbscan_', 'DBSCAN + ') for k in opcoes_disponiveis}

            algoritmo_labels_chave = st.selectbox(
                "Escolha o Algoritmo/Embedding para ver as partições",
                options=opcoes_disponiveis,
                format_func=lambda x: nomes_formatados.get(x, x),
                key=f"sb_menu_{base_selecionada}"
            )

            textos = dados_da_base['textos_originais']
            clusters = dados_da_base[algoritmo_labels_chave]

            # Captura de clusters protegida contra falhas de conversão de dados do DBSCAN modificado
            lista_clusters = []
            for c in clusters:
                try:
                    if c is not None and not pd.isna(c) and str(c).replace('.','').replace('-','').isdigit():
                        v_int = int(float(c))
                        if v_int not in lista_clusters:
                            lista_clusters.append(v_int)
                except:
                    continue
            lista_clusters = sorted(lista_clusters)

            cluster_selecionado = None
            if len(lista_clusters) > 1:
                cluster_selecionado = st.slider(
                    "Selecione o número do Cluster para inspecionar",
                    min_value=int(min(lista_clusters)),
                    max_value=int(max(lista_clusters)),
                    value=int(min(lista_clusters)),
                    key=f"sl_menu_{base_selecionada}"
                )
            elif len(lista_clusters) == 1:
                cluster_selecionado = lista_clusters[0]
                st.info(f"💡 Este algoritmo gerou apenas um grupo válido: **Cluster {cluster_selecionado}**")
            else:
                st.warning("⚠️ Nenhum cluster válido detectado para essa combinação.")

            if cluster_selecionado is not None:
                textos_filtrados = []
                for i, c in enumerate(clusters):
                    try:
                        if c is not None and not pd.isna(c):
                            if int(float(c)) == int(cluster_selecionado):
                                textos_filtrados.append(textos[i])
                    except:
                        continue

                st.write(f"📝 Mostrando os primeiros 10 de {len(textos_filtrados)} textos encontrados no **Cluster {cluster_selecionado}**:")
                if len(textos_filtrados) > 0:
                    for idx, txt in enumerate(textos_filtrados[:10]):
                        st.info(f"**Documento #{idx+1}:** {txt[:300]}...")
                else:
                    st.write("Nenhum documento encontrado para este ID de cluster.")

